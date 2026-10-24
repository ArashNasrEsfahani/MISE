"""Step 5 / A2 - consolidation (docs/step5-plan.md section 2, rules R1-R6 and R9; Astra A2 review in
sim/step5/_astra/A2-rules.txt).

`consolidate(ltm, wm)` turns ONE complete, valid PRACTICE episode's working memory into long-term claims with exactly
one `Store.commit` (episode receipt + clock + claims + one evidence row per claim mutation + write_log, atomic). Test /
probe episodes, incomplete episodes and instrumentation-invalid episodes write nothing (and do not advance the clock).
A retry of an already-consolidated episode is a no-op (durable replay id `cons:<episode>`).

What is kept ("don't save every interaction, that's not how humans work"):
  R1  claims only for USEFUL types (public pickupable ontology, `step5/lexicon/useful_types.json`) plus this task's targets
  R2  one location claim per (type, canonical place) from the track's FINAL state; a pickup supersedes it, a put is a
      placement (verified if re-sighted, else actuator), a drop is an uncertain placement; held tracks give nothing
  R3  natural discovery: the first eligible sighting of a track BEFORE any agent intervention is one search trial per
      configuration unit U, even if the object is picked up later (chronological: later interventions only block LATER trials)
  R3a self-placement (put, drop, ambiguous reacquisition near an own placement) never creates a natural trial
  R4  qualified negative: only after a contemporaneous open, verdict 'visible', enough support / thirds / fraction, in
      range, and a detector capable for that type; stored as "T not detected in P after a qualified inspection", never
      "P is empty"; partial / none inspections are NOT stored; agent-disturbed places are excluded
  R5  placements live in their own table and never touch habits
  R6  contradiction: one per claim per episode (qualified absence or a track-level contradiction, whichever came first);
      support resets the count; a natural positive elsewhere is NOT displacement
  R9  evidence: one row per claim mutation (create, support, contradict, supersede, merge, unmerge ...)
R7 (procedure admission) and R8 (lessons) belong to stage A5 and plug in through `extensions`; here only the bookkeeping
`goal_without_plan` is counted.

Nothing is hard-coded: every number comes from the registry (`ltm.params`, sim/step5/params.json).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

try:
    from ltm_events import ev_hash
    from ltm_habits import alias_map, canon_place, collapse_trials
    from ltm_identity import PlaceBook, dist3, id_key
    from ltm_store import sha256_hex
except ImportError:                     # pragma: no cover
    from sim.ltm_events import ev_hash
    from sim.ltm_habits import alias_map, canon_place, collapse_trials
    from sim.ltm_identity import PlaceBook, dist3, id_key
    from sim.ltm_store import sha256_hex

USEFUL_PATH = Path(__file__).resolve().parent / "step5" / "lexicon" / "useful_types.json"
NATURAL_BASIS = ("inside_inspection", "depth_consistent")
NO_RTYPE = (None, "none", "?", "")


def H(*parts) -> str:
    return sha256_hex("|".join(str(p) for p in parts))[:16]


def load_useful_types(path=None) -> frozenset:
    d = json.loads(Path(path or USEFUL_PATH).read_text(encoding="utf-8"))
    return frozenset(d["useful_types"])


def capable(params: dict, detector_rev, otype: str) -> bool:
    """A negative about `otype` is only meaningful for a type the detector revision usually finds. Unknown -> False."""
    vals = (params.get("detector_capability") or {}).get("values") or {}
    entry = (vals.get(detector_rev) or {}).get(otype)
    if entry is None:
        return False
    recall = entry.get("recall") if isinstance(entry, dict) else entry
    try:
        return float(recall) >= float(params["inspect"]["min_type_recall"]["value"])
    except (TypeError, ValueError):
        return False


def inspection_coverage(params: dict, e: dict, contemporaneous: bool) -> str:
    """'qualified' | 'partial' | 'none' for one inspection event (spec 1.5). Only a qualified inspection can support a
    negative; 'partial' means 'visit again later' and is never stored; everything else is nothing."""
    v = e.get("verdict")
    if v == "partly":
        return "partial"
    if v != "visible":
        return "none"
    if not contemporaneous:
        return "none"
    if any(e.get(f) is None for f in ("frac", "thirds", "support", "range_m")):
        return "none"
    ok = (e["frac"] >= float(_v(params, "inspect", "vis_ok"))
          and e["thirds"] >= int(_v(params, "inspect", "min_thirds"))
          and e["support"] >= int(_v(params, "inspect", "min_support"))
          and e["range_m"] <= float(_v(params, "inspect", "max_range_m")))
    return "qualified" if ok else "none"


def _v(params, section, key):
    return params[section][key]["value"]


def _xyz_of(e):
    return e["xyz"] if e.get("xyz_ok", True) and e.get("xyz") else e.get("in_xyz")


# ---------------------------------------------------------------------------------------------- the pass
class _Pass:
    """Chronological scan of one episode. Produces plain python structures; no database writes."""

    def __init__(self, ltm, wm, useful):
        self.ltm, self.wm, self.params = ltm, wm, ltm.params
        self.kitchen, self.epoch, self.frame = wm.kitchen, wm.epoch, wm.frame
        store = ltm.store
        places = store.select("place_instance", where={"kitchen": self.kitchen, "frame": self.frame})
        pids = {r["place"] for r in places}
        alias = {k: v for k, v in alias_map(store).items() if k in pids}
        self.book = PlaceBook(self.params, places, alias)
        self.all_place_ids = [r["place"] for r in store.select("place_instance")]
        self.min_conf = float(_v(self.params, "consolidate", "min_conf"))
        self.useful, self.targets = useful, wm.targets()
        self.tracks: dict = {}
        self.closed_after: dict = {}
        self.opened_by_me: list = []
        self.disturbed: set = set()
        self.placed_marks: dict = {}
        self.first_nat: list = []          # {otype, place, rtype, ev, track}
        self.neg_cands: list = []          # qualified negatives
        self.placements: list = []         # {otype, place|None, xyz, prov, ev, track}
        self.picked: list = []             # (otype, place) removed by the agent
        self.contra: list = []             # (otype, place, ev, track)
        self.merge_log: list = []
        self.rec_place: dict = {}
        self.stats = dict(abstained=0, ineligible_types=0, deferred_inspections=0, unqualified_inspections=0,
                          low_conf=0, merges_refused=0)
        self.opens: dict = {}

    # -- helpers --------------------------------------------------------------------------------------------
    def cp(self, pid):
        return self.book.canonical(pid)

    def eligible_type(self, otype):
        return otype in self.useful or otype in self.targets

    def src(self, *evs):
        return [ev_hash(self.wm.episode, e) for e in dict.fromkeys(evs)][:16]

    # -- identity pre-pass (receptacle sightings are resolved per view, then merges are judged) ----------------
    def identity_prepass(self):
        """Boxes are resolved per view (deterministic, same-view exclusion). A box that `covers` other centres is NOT a
        new place: it is evidence about duplicates, so it never creates a place and never takes one."""
        views: dict = {}
        order = []
        spanning = []
        for e in self.wm.events:
            if e["kind"] != "receptacle_seen":
                continue
            if e.get("covers"):
                spanning.append(e)
                continue
            vk = e.get("view") or f"solo:{e['ev']}"
            if vk not in views:
                views[vk] = []
                order.append(vk)
            views[vk].append((e["ev"], e["rtype"], e["xyz"]))
        for vk in order:
            for ev, (kind, pid) in self.book.resolve_view(views[vk]).items():
                self.rec_place[ev] = (kind, pid)
                if kind == "abstain":
                    self.stats["abstained"] += 1
        for e in spanning:
            members = []
            for c in e["covers"] + [e["xyz"]]:
                cands = self.book.candidates(e["rtype"], c)
                if cands:
                    members.append(self.cp(cands[0][1]))
            members = sorted(set(members), key=id_key)
            for other in members[1:]:
                if self.cp(other) == self.cp(members[0]):
                    continue
                ok, why = self.book.try_merge(members[0], other, covered_by_one_box=True)
                if ok:
                    self.merge_log.append((self.book.merges[-1], e["ev"]))
                else:
                    self.stats["merges_refused"] += 1

    # -- chronological scan -----------------------------------------------------------------------------------
    def run(self):
        self.identity_prepass()
        for e in self.wm.events:
            getattr(self, "on_" + e["kind"], lambda e: None)(e)
        self.finish_opens()

    def place_for(self, rtype, xyz, create=True):
        if rtype in NO_RTYPE or xyz is None:
            return None
        kind, pid, _ = self.book.resolve(rtype, xyz, create=create)
        if kind == "abstain":
            self.stats["abstained"] += 1
        return pid

    def on_open(self, e):
        pid = self.place_for(e["rtype"], e["xyz"])
        ok = bool(e.get("ok")) or e.get("state_seen") == "open"
        self.opens[e["ev"]] = (pid, ok)
        if e.get("ok") and e.get("by_me", True) and pid:
            self.opened_by_me.append(pid)

    def on_close(self, e):
        pid = self.place_for(e["rtype"], e["xyz"])
        if pid:
            self.closed_after[self.cp(pid)] = e["ev"]

    def finish_opens(self):
        """Distinct fronts opened in one episode are separate places (separation witness, never a merge)."""
        seen = sorted(set(self.cp(p) for p in self.opened_by_me), key=id_key)
        for i, a in enumerate(seen):
            for b in seen[i + 1:]:
                pa, pb = self.book.places[a], self.book.places[b]
                if pa.rtype == pb.rtype and dist3(pa.c, pb.c) <= self.book.near_limit(pa.rtype):
                    self.book.add_separation(a, b, "opened_separately")

    def on_inspection(self, e):
        pid = self.place_for(e["rtype"], e["xyz"])
        if pid is None:
            return
        open_ev = e.get("open_ev")
        op = self.opens.get(open_ev) if open_ev is not None else None
        contemporaneous = bool(op and op[1] and self.cp(op[0]) == self.cp(pid)
                               and not (self.closed_after.get(self.cp(pid), -1) > open_ev))
        cov = inspection_coverage(self.params, e, contemporaneous)
        if cov == "partial":
            self.stats["deferred_inspections"] += 1
            return
        if cov != "qualified":
            self.stats["unqualified_inspections"] += 1
            return
        detected = set(e.get("detected") or [])
        for t in sorted(set(e.get("targets") or [])):
            if t in detected or not capable(self.params, e.get("detector_rev"), t):
                continue
            if (t, self.cp(pid)) in self.disturbed:
                continue
            self.neg_cands.append({"otype": t, "place": pid, "ev": e["ev"], "rev": e.get("detector_rev") or "unknown",
                                   "support": int(e["support"])})

    def track(self, tid, otype):
        return self.tracks.setdefault(tid, {"otype": otype, "views_at": {}, "loc": None, "first_nat": False,
                                            "touched": False, "held": False, "evs": []})

    def near_placed(self, otype, xyz):
        gate = float(_v(self.params, "identity", "placed_reacquire_m"))
        return xyz is not None and any(dist3(xyz, m) <= gate for m in self.placed_marks.get(otype, ()))

    def on_sighting(self, e):
        otype = e["otype"]
        if not self.eligible_type(otype):
            self.stats["ineligible_types"] += 1
            return
        tr = self.track(e["track"], otype)
        tr["evs"].append(e["ev"])
        if tr["held"]:
            return                                              # an object in the hand has no location
        conf = float(e.get("conf") if e.get("conf") is not None else 0.0)
        if conf < self.min_conf or not e.get("xyz_ok", True):
            self.stats["low_conf"] += 1
            return
        basis = e.get("in_basis") or "none"
        place = None
        if e.get("in_rtype") not in NO_RTYPE and e.get("in_xyz") and basis != "none":
            place = self.place_for(e["in_rtype"], e["in_xyz"])
        if place is None:
            return
        cp = self.cp(place)
        views = tr["views_at"].setdefault(cp, set())
        views.add(e.get("view") or f"ev{e['ev']}")
        xyz = _xyz_of(e)
        if tr["touched"]:                                       # a re-sighting of an object the agent placed
            prov = "agent_placed"
        elif self.near_placed(otype, xyz):                      # ambiguous reacquisition (R3a)
            prov = "agent_placed"
        else:
            prov = "natural"
        old = tr["loc"]
        basis_ok = (basis in NATURAL_BASIS) or bool(old and self.cp(old["place"]) == cp and old["basis_ok"])
        tr["loc"] = {"place": place, "xyz": xyz, "conf": max(conf, old["conf"] if old and self.cp(old["place"]) == cp else 0),
                     "basis": basis, "basis_ok": basis_ok, "prov": prov, "ev": e["ev"]}
        if (prov == "natural" and not tr["first_nat"] and basis in NATURAL_BASIS
                and (otype, cp) not in self.disturbed):
            tr["first_nat"] = True
            self.first_nat.append({"otype": otype, "place": place, "ev": e["ev"], "track": e["track"]})

    def on_pickup(self, e):
        if not e.get("ok"):
            return
        otype = e.get("otype") or (self.tracks.get(e["track"]) or {}).get("otype") or "?"
        if not self.eligible_type(otype):
            return
        tr = self.track(e["track"], otype)
        tr["evs"].append(e["ev"])
        if tr["loc"] is not None:
            cp = self.cp(tr["loc"]["place"])
            self.disturbed.add((otype, cp))
            self.picked.append({"otype": otype, "place": tr["loc"]["place"], "ev": e["ev"]})
        tr["loc"], tr["touched"], tr["held"] = None, True, True

    def _placed(self, e, place, rtype_known):
        otype = e["otype"]
        if not self.eligible_type(otype):
            return
        tr = self.track(e["track"], otype)
        tr["evs"].append(e["ev"])
        tr["touched"], tr["held"] = True, False
        self.placed_marks.setdefault(otype, []).append(e["xyz"])
        if place is not None:
            self.disturbed.add((otype, self.cp(place)))
        resighted = bool(e.get("resighted"))
        self.placements.append({"otype": otype, "place": place, "xyz": e["xyz"],
                                "prov": "verified" if resighted else "actuator", "ev": e["ev"], "track": e["track"]})
        tr["loc"] = ({"place": place, "xyz": e["xyz"], "conf": 1.0, "basis": "none", "basis_ok": True,
                      "prov": "agent_placed", "ev": e["ev"]} if (resighted and place is not None) else None)

    def on_put(self, e):
        if not e.get("ok"):
            return
        self._placed(e, self.place_for(e.get("rtype"), e["xyz"]), True)

    def on_drop(self, e):
        if not e.get("ok"):
            return
        self._placed(e, None, False)

    def on_contradiction(self, e):
        tr = self.tracks.get(e["track"])
        if tr is None or tr["loc"] is None or (e.get("views") is not None and e["views"] < 2):
            return
        self.contra.append({"otype": tr["otype"], "place": tr["loc"]["place"], "ev": e["ev"], "track": e["track"],
                            "src": e.get("src") or []})
        tr["loc"] = None


# ---------------------------------------------------------------------------------------------- consolidate
def consolidate(ltm, wm, admission=None, extensions=(), useful=None) -> dict:
    """Consolidate one episode. Returns {status, ...}; statuses that wrote nothing: skipped_mode, incomplete, invalid,
    frozen, epoch_mismatch, kitchen_mismatch; 'duplicate' for an already-consolidated episode; else 'committed'."""
    if getattr(ltm, "frozen", False):
        return {"status": "frozen"}
    if wm.kitchen is not None and wm.kitchen != ltm.kitchen:
        return {"status": "kitchen_mismatch"}
    if wm.mode != "practice":
        return {"status": "skipped_mode", "mode": wm.mode}
    if not wm.complete:
        return {"status": "incomplete"}
    if wm.invalid:
        return {"status": "invalid"}
    store = ltm.store
    event_id = f"cons:{wm.episode}"
    prior = store.select("commit_receipt", where={"event_id": event_id})
    if prior:
        return {"status": "duplicate", "E": prior[0]["E"], "event_id": event_id}
    if wm.epoch != ltm.current_epoch():
        return {"status": "epoch_mismatch", "wm_epoch": wm.epoch, "ltm_epoch": ltm.current_epoch()}
    useful = useful if useful is not None else load_useful_types()
    E0 = store.clock()
    px = _Pass(ltm, wm, useful)
    px.run()
    b, p, k, epoch, frame = px.book, ltm.params, ltm.kitchen, wm.epoch, wm.frame

    # ---- mint final place ids (order of creation) ----------------------------------------------------------
    n = PlaceBook.max_final(px.all_place_ids)
    final_of = {}
    for pid in b.provisional():
        n += 1
        final_of[pid] = f"pl:{n}"

    def F(pid):
        return final_of.get(pid, pid)

    def C(pid):                      # canonical, final
        return None if pid is None else F(b.canonical(pid))

    def rtype_of(pid):
        return b.places[b.canonical(pid)].rtype

    # ---- existing LTM state ----------------------------------------------------------------------------------
    ex_loc = {r["claim"]: r for r in store.select("location_claim", where={"kitchen": k})}
    ex_neg = store.select("negative_evidence", where={"kitchen": k})
    ex_plc = {r["placement"]: r for r in store.select("placement", where={"kitchen": k})}
    ex_trials = store.select("search_trial", where={"kitchen": k})
    ex_trial_ids = {r["u"] for r in ex_trials}
    alias_final = {F(a): F(c) for a, c in b.alias.items()}
    ex_collapsed = collapse_trials(ex_trials, alias_final)
    pos_ex = {(t["epoch"], t["otype"], t["place"]) for t in ex_collapsed if t["outcome"] == "positive"}
    habit_ids = {r["habit"] for r in store.select("habit", where={"kitchen": k})}

    ops, stats = [], px.stats
    for key in ("claims_created", "claims_supported", "claims_contradicted", "claims_retired", "negatives_written",
                "negatives_superseded", "trials_pos", "trials_neg", "places_created", "places_updated", "merges",
                "surprises", "placements", "goal_without_plan"):
        stats[key] = 0

    def claim_id(otype, cp):
        return "lc:" + H(k, frame, otype, cp)

    # ---- location claims from final track states -------------------------------------------------------------
    loc_final: dict = {}
    for tid in sorted(px.tracks):
        tr = px.tracks[tid]
        lc = tr["loc"]
        if tr["held"] or lc is None or lc["place"] is None:
            continue
        key = (tr["otype"], C(lc["place"]))
        views = len(tr["views_at"].get(b.canonical(lc["place"]), ()))
        verified = lc["basis_ok"] or views >= 2
        cur = loc_final.get(key)
        cand = {"otype": tr["otype"], "place": key[1], "xyz": [round(x, 3) for x in lc["xyz"]], "conf": lc["conf"],
                "verified": "verified" if verified else "trusted", "prov": lc["prov"], "evs": list(tr["evs"]),
                "views": views, "basis": lc["basis"], "n_tracks": 1}
        if cur is None:
            loc_final[key] = cand
        else:
            cur["n_tracks"] += 1
            cur["evs"] += cand["evs"]
            if cand["verified"] == "verified":
                cur["verified"] = "verified"
            if cand["prov"] == "natural":
                cur["prov"] = "natural"
            if cand["conf"] > cur["conf"]:
                cur["conf"], cur["xyz"] = cand["conf"], cand["xyz"]
            cur["views"] = max(cur["views"], cand["views"])

    # ---- placements -------------------------------------------------------------------------------------------
    plc_final: dict = {}
    for pm in px.placements:
        place = C(pm["place"]) if pm["place"] is not None else None
        if place is None:                                       # a drop: take the place from the track's re-sighting
            lc = px.tracks[pm["track"]]["loc"]
            place = C(lc["place"]) if lc and lc["place"] else None
        gid = "pm:" + H(k, epoch, pm["otype"], place if place else f"{round(pm['xyz'][0], 1)},{round(pm['xyz'][2], 1)}")
        cur = plc_final.get(gid)
        if cur is None or pm["prov"] == "verified":
            plc_final[gid] = {"otype": pm["otype"], "place": place, "xyz": [round(x, 3) for x in pm["xyz"]],
                              "prov": pm["prov"], "evs": [pm["ev"]]}
        else:
            cur["evs"].append(pm["ev"])

    # ---- contradictions (one per claim per episode) --------------------------------------------------------------
    supported_keys = set(loc_final) | {(v["otype"], v["place"]) for v in plc_final.values() if v["place"]}
    contra_src: dict = {}                       # claim id -> (source label, src evs)
    qualified_neg = []
    for nc in px.neg_cands:
        key = (nc["otype"], C(nc["place"]))
        if key in supported_keys:
            continue                                            # positive evidence in the same episode wins
        qualified_neg.append((nc, key))
    abs_keys = [(key, "qualified_absence", [nc["ev"]]) for nc, key in qualified_neg]
    for c in px.contra:
        key = (c["otype"], C(c["place"]))
        if key in supported_keys:
            continue                                            # another object of that type is still there
        abs_keys.append((key, "track_contradiction", [c["ev"]]))
    for key, source, evs in abs_keys:
        for cid, row in sorted(ex_loc.items()):
            if row["otype"] == key[0] and row["status"] != "retired" and C(row["place"]) == key[1] and cid not in contra_src:
                contra_src[cid] = (source, evs)

    # ---- trials ---------------------------------------------------------------------------------------------------
    new_pos: dict = {}
    for fn in px.first_nat:
        key = (epoch, fn["otype"], C(fn["place"]))
        if key in pos_ex or key in new_pos:
            continue
        new_pos[key] = fn
    new_neg: dict = {}
    for nc, key in qualified_neg:
        tkey = (epoch, key[0], key[1])
        if tkey in pos_ex or tkey in new_pos or tkey in new_neg:
            continue
        new_neg[tkey] = nc

    # ================================================================== build ops (final ids from here on) ======
    # places
    for pid in b.provisional():
        pl = b.places[pid]
        ext = dict(pl.extent)
        if pl.sep:
            ext["sep"] = sorted((F(x) for x in pl.sep), key=id_key)
        ops.append({"table": "place_instance", "op": "insert",
                    "values": {"place": F(pid), "kitchen": k, "frame": frame, "epoch": epoch, "rtype": pl.rtype,
                               "cx": round(pl.c[0], 4), "cy": round(pl.c[1], 4), "cz": round(pl.c[2], 4),
                               "extent": ext, "created_E": "$E"},
                    "evidence": {"rule": "R-ID", "mutation": "create", "summary": {"rtype": pl.rtype},
                                 "src_ev": px.src(*[e["ev"] for e in wm.events if e["kind"] == "receptacle_seen"][:3])}})
        stats["places_created"] += 1
    for pid, pl in sorted(b.places.items(), key=lambda t: id_key(t[0])):
        if pl.new or not pl.dirty:
            continue
        ext = dict(pl.extent)
        if pl.sep:
            ext["sep"] = sorted((F(x) for x in pl.sep), key=id_key)
        ops.append({"table": "place_instance", "op": "update", "key": {"place": pid}, "values": {"extent": ext},
                    "evidence": {"rule": "R-ID", "mutation": "support", "summary": {"rtype": pl.rtype, "what": "geometry_or_witness"},
                                 "src_ev": []}})
        stats["places_updated"] += 1
    for (loser, winner), mev in px.merge_log:
        ops.append({"table": "place_alias", "op": "upsert",
                    "values": {"place": F(loser), "canonical": F(winner), "active": True, "E": "$E"},
                    "evidence": {"rule": "R-ID", "mutation": "merge", "summary": {"loser": F(loser), "winner": F(winner)},
                                 "src_ev": px.src(mev)}})
        stats["merges"] += 1

    # negatives superseded by a positive at the same canonical place
    pos_units = {key for key in loc_final} | {(v["otype"], v["place"]) for v in plc_final.values() if v["place"]}
    for r in sorted(ex_neg, key=lambda r: r["neg"]):
        if r["superseded"]:
            continue
        key = (r["otype"], C(r["place"]))
        if key in pos_units:
            by = claim_id(*key) if key in loc_final else "pm:" + H(k, epoch, *key)
            ops.append({"table": "negative_evidence", "op": "update", "key": {"neg": r["neg"]},
                        "values": {"superseded": True, "superseded_by": by},
                        "evidence": {"rule": "R4", "mutation": "supersede",
                                     "summary": {"otype": r["otype"], "place": key[1], "by": "later_positive"}, "src_ev": []}})
            stats["negatives_superseded"] += 1
            stats["surprises"] += 1

    # location claims
    for key in sorted(loc_final, key=lambda t: (t[0], id_key(t[1]))):
        v = loc_final[key]
        cid = claim_id(*key)
        old = ex_loc.get(cid)
        verified = "verified" if (v["verified"] == "verified" or (old and old["verified"] == "verified"
                                                                  and old["status"] != "retired")) else "trusted"
        mutation = "create" if old is None else ("reactivate" if old["status"] == "retired" else "support")
        values = {"claim": cid, "kitchen": k, "epoch": epoch, "frame": frame, "otype": key[0], "place": key[1],
                  "xyz": v["xyz"], "verified": verified, "provenance": v["prov"], "conf": round(v["conf"], 3),
                  "E_support": "$E", "contradictions": 0, "status": "current"}
        if old is None:
            values["E_created"] = "$E"
        ops.append({"table": "location_claim", "op": "upsert", "values": values,
                    "evidence": {"rule": "R2", "mutation": mutation,
                                 "summary": {"otype": key[0], "place": key[1], "basis": v["basis"], "views": v["views"],
                                             "provenance": v["prov"], "n_tracks": v["n_tracks"]},
                                 "src_ev": px.src(*v["evs"]), "conf": round(v["conf"], 3)}})
        stats["claims_created" if old is None else "claims_supported"] += 1

    # pickup supersedes a location (only if nothing of that type was re-supported at the place)
    seen_retire = set()
    for pk in px.picked:
        key = (pk["otype"], C(pk["place"]))
        cid = claim_id(*key)
        old = ex_loc.get(cid)
        if key in loc_final or old is None or old["status"] == "retired" or cid in seen_retire:
            continue
        seen_retire.add(cid)
        ops.append({"table": "location_claim", "op": "update", "key": {"claim": cid}, "values": {"status": "retired"},
                    "evidence": {"rule": "R2", "mutation": "supersede",
                                 "summary": {"otype": key[0], "place": key[1], "by": "pickup"}, "src_ev": px.src(pk["ev"])}})
        stats["claims_retired"] += 1

    # contradictions
    for cid in sorted(contra_src):
        old = ex_loc[cid]
        if cid in seen_retire:
            continue
        source, evs = contra_src[cid]
        ops.append({"table": "location_claim", "op": "update", "key": {"claim": cid},
                    "values": {"contradictions": int(old["contradictions"]) + 1},
                    "evidence": {"rule": "R6", "mutation": "contradict",
                                 "summary": {"otype": old["otype"], "place": old["place"], "source": source,
                                             "c": int(old["contradictions"]) + 1},
                                 "src_ev": px.src(*evs)}})
        stats["claims_contradicted"] += 1
        stats["surprises"] += 1

    # placements (own table; never touches habits)
    for gid in sorted(plc_final):
        v = plc_final[gid]
        old = ex_plc.get(gid)
        ops.append({"table": "placement", "op": "upsert",
                    "values": {"placement": gid, "kitchen": k, "epoch": epoch, "place": v["place"], "otype": v["otype"],
                               "xyz": v["xyz"], "provenance": v["prov"], "E": "$E", "status": "current"},
                    "evidence": {"rule": "R5", "mutation": "create" if old is None else "support",
                                 "summary": {"otype": v["otype"], "place": v["place"], "provenance": v["prov"]},
                                 "src_ev": px.src(*v["evs"])}})
        stats["placements"] += 1
    # a qualified negative at a place holding the agent's own placement makes that placement historical
    for nc, key in qualified_neg:
        for gid, row in sorted(ex_plc.items()):
            if row["otype"] == key[0] and row["status"] == "current" and row["place"] and C(row["place"]) == key[1] \
                    and gid not in plc_final:
                ops.append({"table": "placement", "op": "update", "key": {"placement": gid}, "values": {"status": "historical"},
                            "evidence": {"rule": "R6", "mutation": "contradict",
                                         "summary": {"otype": key[0], "place": key[1], "source": "qualified_absence"},
                                         "src_ev": px.src(nc["ev"])}})

    # qualified negatives (never "the place is empty")
    done_neg = set()
    for nc, key in qualified_neg:
        nid = "ng:" + H(k, epoch, key[0], key[1])
        if nid in done_neg:
            continue
        done_neg.add(nid)
        old = next((r for r in ex_neg if r["neg"] == nid), None)
        ops.append({"table": "negative_evidence", "op": "upsert",
                    "values": {"neg": nid, "kitchen": k, "epoch": epoch, "place": key[1], "otype": key[0],
                               "detector_rev": nc["rev"], "support": nc["support"], "E": "$E", "superseded": False},
                    "evidence": {"rule": "R4", "mutation": "create" if old is None else "support",
                                 "summary": {"otype": key[0], "place": key[1], "detector_rev": nc["rev"],
                                             "kind": "not_detected_after_qualified_inspection"},
                                 "src_ev": px.src(nc["ev"])}})
        stats["negatives_written"] += 1

    # search trials (+ habit definition rows)
    habit_seen = set()

    def habit_ops(otype, rtype, ev_src):
        for level, active in (("kitchen", True), ("general", False)):
            hid = f"hb:{'k' if level == 'kitchen' else 'g'}:{k}:{otype}:{rtype}"
            if hid in habit_seen:
                continue
            habit_seen.add(hid)
            ops.append({"table": "habit", "op": "upsert",
                        "values": {"habit": hid, "kitchen": k, "level": level, "otype": otype, "rtype": rtype,
                                   "active": active and True, "E_last": "$E"},
                        "evidence": {"rule": "R3", "mutation": "create" if hid not in habit_ids else "support",
                                     "summary": {"otype": otype, "rtype": rtype, "level": level}, "src_ev": ev_src}})

    for (ep, otype, cp), fn in sorted(new_pos.items(), key=lambda t: (t[0][1], id_key(t[0][2]))):
        u = "tp:" + H(k, ep, otype, cp, "p")
        if u in ex_trial_ids:
            continue
        rt = rtype_of(fn["place"])
        ops.append({"table": "search_trial", "op": "insert",
                    "values": {"u": u, "kitchen": k, "epoch": ep, "otype": otype, "place": cp, "rtype": rt,
                               "outcome": "positive", "E": "$E"},
                    "evidence": {"rule": "R3", "mutation": "create",
                                 "summary": {"otype": otype, "place": cp, "outcome": "positive"}, "src_ev": px.src(fn["ev"])}})
        habit_ops(otype, rt, px.src(fn["ev"]))
        stats["trials_pos"] += 1
    for (ep, otype, cp), nc in sorted(new_neg.items(), key=lambda t: (t[0][1], id_key(t[0][2]))):
        u = "tn:" + H(k, ep, otype, cp, "n")
        if u in ex_trial_ids:
            continue
        rt = rtype_of(nc["place"])
        ops.append({"table": "search_trial", "op": "insert",
                    "values": {"u": u, "kitchen": k, "epoch": ep, "otype": otype, "place": cp, "rtype": rt,
                               "outcome": "negative", "E": "$E"},
                    "evidence": {"rule": "R4", "mutation": "create",
                                 "summary": {"otype": otype, "place": cp, "outcome": "negative"}, "src_ev": px.src(nc["ev"])}})
        habit_ops(otype, rt, px.src(nc["ev"]))
        stats["trials_neg"] += 1

    # ---- R7 bookkeeping, episode receipt, extensions ----------------------------------------------------------------
    term = wm.terminal
    adm = wm.admission
    goal_ok = adm["goal_ok"] if adm else (admission if isinstance(admission, bool) else None)
    if goal_ok and not term.get("exec_ok"):
        stats["goal_without_plan"] = 1
    ctx = SimpleNamespace(ltm=ltm, wm=wm, kitchen=k, epoch=epoch, frame=frame, final_of=final_of, F=F, C=C,
                          goal_ok=goal_ok, exec_ok=bool(term.get("exec_ok")), E_next=E0 + 1)
    for ext in extensions:
        ops.extend(ext(ctx) or [])
    stat_json = {key: int(v) for key, v in stats.items()}
    ops.append({"table": "episodes", "op": "insert",
                "values": {"episode": wm.episode, "kitchen": k, "mode": "practice", "world": wm.world, "epoch": epoch,
                           "level": wm.level, "lang": wm.lang, "E": "$E", "exec_ok": bool(term.get("exec_ok")),
                           "goal_ok": goal_ok, "invalid": False, "stats": stat_json}})
    res = store.commit(event_id, "consolidation", ops, episode=wm.episode, advance_clock=True, expect_clock=E0)
    res["stats"] = stat_json
    res["minted"] = dict(final_of)
    res["status"] = res.get("status", "committed")
    return res


def unmerge(ltm, place: str, reason: str = "manual_correction") -> dict:
    """Undo a (wrong) merge: the alias goes inactive, both places get a separation witness (so it is never merged again
    by the same kind of evidence), and every claim / trial keeps its original place id, so nothing is lost and read-time
    collapse stops double counting nothing. The history (evidence + write_log) is kept."""
    store, k = ltm.store, ltm.kitchen
    if getattr(ltm, "frozen", False):
        return {"status": "frozen"}
    rows = store.select("place_alias", where={"place": place})
    if not rows or not rows[0]["active"]:
        return {"status": "no_active_alias"}
    winner = rows[0]["canonical"]
    E = store.clock()
    ops = [{"table": "place_alias", "op": "update", "key": {"place": place}, "values": {"active": False, "E": E},
            "evidence": {"rule": "R-ID", "mutation": "unmerge", "summary": {"loser": place, "winner": winner,
                                                                           "reason": reason}, "src_ev": []}}]
    for a, b in ((place, winner), (winner, place)):
        r = store.select("place_instance", where={"place": a})[0]
        ext = dict(r["extent"] or {})
        ext["sep"] = sorted(set(ext.get("sep", [])) | {b}, key=id_key)
        ops.append({"table": "place_instance", "op": "update", "key": {"place": a}, "values": {"extent": ext},
                    "evidence": {"rule": "R-ID", "mutation": "unmerge", "summary": {"witness": b}, "src_ev": []}})
    return store.commit(f"unmerge:{place}:{winner}:{E}", "admin", ops, stamp=E)
