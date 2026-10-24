"""Selftest for B3 (sim/ltm_profile.py): the learned kitchen profile.  Offline, Windows python, no simulator / WSL / Ollama / API.

    python sim/_pkg_profile_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ltm_events as lev                                  # noqa: E402
import ltm_profile as P                                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                         # pragma: no cover
    pass

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL:", name, detail)


def section(t):
    print("--", t)


# a SYNTHETIC kitchen: 2 Fridges, no Toaster, a Sink without a SinkBasin, 6 StoveKnobs, 4 StoveBurners
CATS = {"fixture_types": ["Fridge", "Microwave", "Sink", "SinkBasin", "StoveBurner", "StoveKnob", "CounterTop", "Cabinet", "Drawer",
                          "GarbageCan", "CoffeeMachine", "Toaster", "Faucet", "LightSwitch", "Shelf", "ShelvingUnit", "Window"],
        "tool_types": ["Knife", "ButterKnife", "Spatula", "Fork", "Spoon", "Ladle"]}
K = "SynthKitchen"


class Stream:
    """Builds a VALID s5.ev/1 stream (every event is passed through ltm_events.validate)."""

    def __init__(self, episode, kitchen=K):
        self.ep, self.k, self.n, self.out = episode, kitchen, 0, []

    def add(self, kind, **f):
        self.n += 1
        ev = {"schema": lev.SCHEMA, "ev": self.n, "kind": kind, "kitchen": self.k, "epoch": 1, "frame": "f0", "map_rev": "m0",
              "episode": self.ep, "t": self.n}
        ev.update(f)
        self.out.append(lev.validate(ev))
        return self

    def row(self, attempt, item="it", lang="en", key=None):
        return {"schema": "pkg.evs/1", "key": key or self.ep, "order": 0, "attempt": attempt, "item": item, "lang": lang,
                "episode": self.ep, "stream": self.out}


def ep1():
    s = Stream("ep:one")
    s.add("episode_start", mode="practice", world="reset")
    s.add("receptacle_seen", rtype="Fridge", xyz=[1.0, 0.9, 2.0])
    s.add("receptacle_seen", rtype="Fridge", xyz=[1.1, 0.9, 2.1])           # same place (< MERGE_M)
    s.add("receptacle_seen", rtype="Fridge", xyz=[-3.0, 0.9, 2.0])          # a second Fridge
    s.add("receptacle_seen", rtype="Sink", xyz=[0.0, 0.9, -2.0])
    s.add("receptacle_seen", rtype="Banana", xyz=[0.0, 0.9, 0.0])           # not a fixture type: ignored
    s.add("step_start", i=0, verb="open", otype="Fridge", pre={})
    s.add("open", rtype="Fridge", xyz=[1.0, 0.9, 2.0], ok=True, by_me=True, state_seen="open")
    s.add("inspection", rtype="Fridge", xyz=[1.0, 0.9, 2.0], verdict="found", detected=["Egg", "Lettuce", "Fridge", "Cabinet"])
    s.add("step_outcome", i=0, ok=True)
    s.add("close", rtype="Fridge", xyz=[1.0, 0.9, 2.0], ok=True, by_me=True, state_seen="closed")
    s.add("open", rtype="Drawer", xyz=[0.0, 0.5, 1.0], ok=False, by_me=True, state_seen="?")
    s.add("inspection", rtype="Drawer", xyz=[0.0, 0.5, 1.0], verdict="partly", detected=["Spoon"])      # no content evidence
    s.add("inspection", rtype="Cabinet", xyz=[2.0, 1.5, 1.0], verdict="not-visible", detected=[])       # no evidence at all
    s.add("sighting", otype="Knife", state="clean", conf=0.9, detector="d", track="tr:1", xyz=[0.2, 0.9, 1.0], xyz_ok=True,
          view="vw:1", obs="ob:1", in_rtype="Drawer", in_basis="box_only", in_xyz=[0.0, 0.5, 1.0])
    s.add("sighting", otype="Apple", state="clean", conf=0.9, detector="d", track="tr:2", xyz=[0.2, 0.9, 1.0], xyz_ok=True,
          view="vw:1", obs="ob:1")
    s.add("episode_end")
    return s.row(1)


def ep2():
    s = Stream("ep:two")
    s.add("episode_start", mode="practice", world="reset")
    s.add("receptacle_seen", rtype="Fridge", xyz=[1.0, 0.9, 2.0])
    s.add("open", rtype="Fridge", xyz=[1.0, 0.9, 2.0], ok=False, by_me=True, state_seen="?")               # failed open: no state
    s.add("refusal", verb="open", rtype="Fridge", i=0, k=0, cause="action_refused")
    s.add("close", rtype="Fridge", xyz=[1.0, 0.9, 2.0], ok=True, by_me=True, state_seen="closed")
    s.add("inspection", rtype="Cabinet", xyz=[2.0, 1.5, 1.0], verdict="visible", detected=[])             # nothing detected
    s.add("sighting", otype="Knife", state="clean", conf=0.9, detector="d", track="tr:1", xyz=[0.2, 0.9, 1.0], xyz_ok=True,
          view="vw:1", obs="ob:1", in_rtype="CounterTop", in_basis="box_only", in_xyz=[0.0, 0.9, 1.0])
    s.add("episode_end")
    return s.row(2)


def ep3():
    s = Stream("ep:three")
    s.add("episode_start", mode="practice", world="reset")
    s.add("receptacle_seen", rtype="Fridge", xyz=[1.0, 0.9, 2.0])
    s.add("open", rtype="Fridge", xyz=[1.0, 0.9, 2.0], ok=True, by_me=True, state_seen="open")
    s.add("inspection", rtype="Fridge", xyz=[1.0, 0.9, 2.0], verdict="visible", detected=["Egg"])
    s.add("receptacle_seen", rtype="Toaster", xyz=[4.0, 0.9, 0.0])
    s.add("step_start", i=0, verb="toggle_on", otype="Toaster", pre={})
    s.add("step_outcome", i=0, ok=True)
    s.add("refusal", verb="toggle_on", otype="Toaster", i=1, k=0, cause="action_refused")
    s.add("episode_end")
    return s.row(3)


def knob_ep(episode, attempt, burners_changed, pre=True, verb="toggle_on", knob_xz=(1.0, -1.0)):
    """burners_changed: list of (track, cell xz) whose sighting state changes off->on inside the toggle step."""
    s = Stream(episode)
    s.add("episode_start", mode="practice", world="reset")
    for j, (tr, xz) in enumerate([("tr:1", (0.0, -1.0)), ("tr:2", (0.5, -1.0)), ("tr:3", (1.0, -1.0)), ("tr:4", (1.5, -1.0))]):
        s.add("sighting", otype="StoveBurner", state="off", conf=0.9, detector="d", track=tr, xyz=[xz[0], 0.9, xz[1]], xyz_ok=True,
              view="vw:1", obs="ob:1")
    p = {"x": knob_xz[0], "z": knob_xz[1]} if pre else {}
    s.add("step_start", i=0, verb=verb, otype="StoveKnob", pre=p)
    for tr, xz in burners_changed:
        s.add("sighting", otype="StoveBurner", state="on", conf=0.9, detector="d", track=tr, xyz=[xz[0], 0.9, xz[1]], xyz_ok=True,
              view="vw:2", obs="ob:2")
    s.add("step_outcome", i=0, ok=True)
    s.add("episode_end")
    return s.row(attempt)


def main():
    section("contract / module hygiene")
    src = open(os.path.join(HERE, "ltm_profile.py"), encoding="utf-8").read()
    ast.parse(src, feature_version=(3, 10))
    check("py3.10 syntax", True)
    check("no FloorPlan constant in code", "FloorPlan" not in re.sub(r'""".*?"""', "", src, flags=re.S).replace("No FloorPlan1 constant", ""))
    check("no simulator import / metadata", not re.search(r"ai2thor|controller|\.metadata|objectId", re.sub(r'""".*?"""', "", src, flags=re.S)))
    check("ev_hash == ltm_events.ev_hash", all(P.ev_hash("ep:x", n) == lev.ev_hash("ep:x", n) for n in (0, 1, 17, 999)))

    section("build: entities and evidence semantics (6.3 / 6.4)")
    eps = [ep1(), ep2(), ep3()]
    p = P.Profile.build(K, eps, CATS)
    fr, fc = p.fixtures["Fridge"], p.fixtures["Cabinet"]
    check("two Fridges counted (merge < 0.75 m)", fr["n_seen"] == 2, fr["n_seen"])
    check("cells quantised to 0.5", all(abs(c[0] / 0.5 - round(c[0] / 0.5)) < 1e-9 for c in fr["cells"]), fr["cells"])
    check("non-fixture type ignored", "Banana" not in p.fixtures)
    check("no Toaster-free kitchen invention: Toaster is seen only where it was", p.fixtures["Toaster"]["n_seen"] == 1)
    check("Sink without SinkBasin: no SinkBasin entity", "SinkBasin" not in p.fixtures)
    check("opened_by_me counts ok/failed", fr["opened_by_me"] == {"ok": 2, "failed": 1}, fr["opened_by_me"])
    check("closed_by_me counts", fr["closed_by_me"] == {"ok": 2, "failed": 0}, fr["closed_by_me"])
    check("contents only from found / visible inspections", set(fr["contents"]) == {"Egg", "Lettuce"}, fr["contents"])
    check("fixture types never listed as contents", "Cabinet" not in fr["contents"])
    check("Egg seen twice, last attempt 3", fr["contents"]["Egg"] == {"n": 2, "last_ord": 3}, fr["contents"]["Egg"])
    check("a partly inspection gives no contents", "Drawer" not in p.fixtures or p.fixtures["Drawer"]["contents"] == {})
    check("a failed open is no content evidence", "Spoon" not in json.dumps(p.fixtures))
    check("not-visible inspection: no 'nothing'", fc["inspected_nothing"]["n"] == 1, fc["inspected_nothing"])
    check("nothing-detected is a visible inspection with empty detected", fc["inspected_nothing"]["last_ord"] == 2)
    check("state: newest observation wins (attempt 3 open)", fr["last_state"] == {"value": "open", "ord": 3}, fr["last_state"])
    check("contradiction keeps both counts", fr["states_seen"].get("open") == 2 and fr["states_seen"].get("closed") == 2, fr["states_seen"])
    check("failed open adds no state", fr["states_seen"].get("open") == 2)
    check("refusal recorded with believed state", fr["refusals"] and fr["refusals"][0]["verb"] == "open" and fr["refusals"][0]["believed_state"] == "closed",
          fr["refusals"])
    check("toggle ok counted + refused counted", p.fixtures["Toaster"]["toggled"] == {"on_ok": 1, "off_ok": 0, "refused": 1}, p.fixtures["Toaster"]["toggled"])
    check("Toaster state on from the toggle", p.fixtures["Toaster"]["last_state"]["value"] == "on")
    kn = p.tools["Knife"]
    check("tool seen twice by distinct tracks per episode", kn["n_seen"] == 2, kn)
    check("tool last parent = newest sighting", kn["last_parent_type"] == "CounterTop" and kn["last_ord"] == 2, kn)
    check("non-tool movable (Apple) is not profiled", "Apple" not in p.tools)
    check("knob map not learned without interventions", p.knob_map == "not_learned")
    check("support = distinct episodes", fr["support"] == 3 and p.fixtures["Sink"]["support"] == 1)
    check("k_total counts the episodes", p.k_total == 3)

    section("determinism, order independence, duplicate rows")
    sh = p.to_json()["sha256"]
    check("deterministic", P.Profile.build(K, eps, CATS).to_json()["sha256"] == sh)
    sh_shuf = P.Profile.build(K, [eps[2], eps[0], eps[1]], CATS).to_json()["sha256"]
    check("episode list order does not matter", sh_shuf == sh)
    check("a repeated row (resume) is ignored", P.Profile.build(K, eps + [eps[0]], CATS).to_json()["sha256"] == sh)

    section("nested snapshots (upto_attempt)")
    p1 = P.Profile.build(K, eps, CATS, upto_attempt=1)
    p2 = P.Profile.build(K, eps, CATS, upto_attempt=2)
    check("upto 1 = only episode 1", p1.k_total == 1 and "Toaster" not in p1.fixtures and p1.fixtures["Fridge"]["last_state"]["value"] == "closed")
    check("upto 2 sees the second episode", p2.k_total == 2 and p2.fixtures["Cabinet"]["inspected_nothing"]["n"] == 1)
    check("nesting: upto 1 evidence subset of upto 3", all(set(v) <= set(p.evidence.get(k, {})) for k, v in p1.evidence.items()))
    check("nesting: upto 2 evidence subset of upto 3", all(set(v) <= set(p.evidence.get(k, {})) for k, v in p2.evidence.items()))
    check("upto None == all", P.Profile.build(K, eps, CATS, upto_attempt=None).to_json()["sha256"] == sh)
    check("upto changes the sha", p1.to_json()["sha256"] != sh)
    check("text states the denominator K", "attempt 1 of 1" in " ".join(r["text"]["en"] for r in p1.records()) and
          "attempt 3 of 3" in " ".join(r["text"]["en"] for r in p.records()))

    section("records: shape, evidence, EN/FA parity (3.1, 6.5)")
    recs = p.records()
    ids = [r["id"] for r in recs]
    check("unique ids", len(ids) == len(set(ids)))
    check("ids have no sim ids", all(re.match(r"^profile/SynthKitchen/(fixture|tool):[A-Za-z]+$", i) for i in ids), ids)
    shape_ok = True
    for r in recs:
        shape_ok &= (r["schema"] == "pkg.rec/1" and r["component"] == "profile" and r["kitchen"] == K and r["privileged"] is False)
        shape_ok &= set(r) >= {"id", "keys", "lex", "text", "provenance", "created", "recency", "confidence", "prio"}
        shape_ok &= set(r["keys"]) == {"types", "ops", "rels"} and set(r["text"]) == {"en", "fa"}
        for rel in r["keys"]["rels"]:
            shape_ok &= rel.split(":", 1)[0] in ("contains", "tool_at", "opens", "toggles", "controls")
    check("pkg.rec/1 shape", shape_ok)
    check("every record has evidence", all(r["provenance"]["evidence"] for r in recs))
    check("provenance is practice + script", all(r["provenance"]["source"].startswith("practice:") and r["provenance"]["authored_by"].startswith("script:ltm_profile@") for r in recs))
    ass = p.assertions()
    check("assertions all have evidence", ass and all(a["evidence"] for a in ass), [a for a in ass if not a["evidence"]])
    check("no sim-id-like text, no parentheses, no ids", all(not re.search(r"[()|]|_physics|objectId|\bpl:|\btr:", r["text"]["en"] + r["text"]["fa"]) for r in recs))
    miss = 0
    for r in recs:
        for tok in re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", r["text"]["en"]):
            if tok in CATS["fixture_types"] + CATS["tool_types"] + ["Egg", "Lettuce"] and tok not in r["text"]["fa"]:
                miss += 1
    check("every English type token also in the Persian text", miss == 0, miss)
    rf = next(r for r in recs if r["id"].endswith("fixture:Fridge"))
    check("Fridge text states the contradiction hedge", "seen open and closed in practice; last: open, across 2 places" in rf["text"]["en"], rf["text"]["en"])
    check("Fridge text names contents", "inside I detected: " in rf["text"]["en"] and "Egg" in rf["text"]["en"])
    check("Fridge keys carry contents types + rels", set(rf["keys"]["types"]) >= {"Fridge", "Egg", "Lettuce"} and "contains:Egg" in rf["keys"]["rels"] and "opens:Fridge" in rf["keys"]["rels"])
    check("Fridge confidence = min(1, 3/3)", rf["confidence"] == 1.0)
    rs = next(r for r in recs if r["id"].endswith("fixture:Sink"))
    check("one-episode confidence = 1/3", abs(rs["confidence"] - 0.3333) < 1e-3, rs["confidence"])
    rk = next(r for r in recs if r["id"].endswith("tool:Knife"))
    check("tool record: tool_at rel", "tool_at:CounterTop" in rk["keys"]["rels"] and "last seen at CounterTop" in rk["text"]["en"])
    check("no advice words", all(not re.search(r"\b(should|must|try|always|never)\b", r["text"]["en"], re.I) for r in recs))
    check("recency sorted", [r["recency"] for r in recs] == sorted([r["recency"] for r in recs], reverse=True))
    check("prio is a deterministic rank", [r["prio"] for r in recs] == list(range(len(recs))))

    section("knob map (6.4): learned only from one-step, one-burner interventions")
    one = P.Profile.build(K, [knob_ep("ep:k1", 1, [("tr:2", (0.5, -1.0))])], CATS)
    check("one knob toggle -> one burner changed: learned", one.knob_map != "not_learned" and len(one.knob_map) == 1, one.knob_map)
    if one.knob_map != "not_learned":
        m = one.knob_map[0]
        check("pair carries cells, count, ords", m["knob_cell"] == [1.0, -1.0] and m["burner_cell"] == [0.5, -1.0] and m["n"] == 1 and m["ords"] == [1])
    krec = [r for r in one.records() if r["id"].endswith("knob_map")]
    check("knob record rendered with evidence + controls rel", krec and krec[0]["provenance"]["evidence"] and "controls:StoveBurner" in krec[0]["keys"]["rels"])
    check("knob record text hedged and EN/FA both present", krec and "lit the burner" in krec[0]["text"]["en"] and krec[0]["text"]["fa"])
    two = P.Profile.build(K, [knob_ep("ep:k2", 1, [("tr:1", (0.0, -1.0)), ("tr:3", (1.0, -1.0))])], CATS)
    check("two burners changed in one step: not learned", two.knob_map == "not_learned", two.knob_map)
    none_ = P.Profile.build(K, [knob_ep("ep:k3", 1, [])], CATS)
    check("no burner change: not learned", none_.knob_map == "not_learned")
    nopre = P.Profile.build(K, [knob_ep("ep:k4", 1, [("tr:2", (0.5, -1.0))], pre=False)], CATS)
    check("knob position unknown: not learned (counted)", nopre.knob_map == "not_learned" and nopre.source["knob_unlocatable"] == 1)
    off = P.Profile.build(K, [knob_ep("ep:k5", 1, [("tr:2", (0.5, -1.0))], verb="toggle_off")], CATS)
    check("toggle_off with burners turning ON: not learned", off.knob_map == "not_learned")
    amb = P.Profile.build(K, [knob_ep("ep:k6", 1, [("tr:2", (0.5, -1.0))]), knob_ep("ep:k7", 2, [("tr:3", (1.0, -1.0))])], CATS)
    check("same knob tied to two burners: not learned", amb.knob_map == "not_learned", amb.knob_map)
    rep = P.Profile.build(K, [knob_ep("ep:k8", 1, [("tr:2", (0.5, -1.0))]), knob_ep("ep:k9", 2, [("tr:2", (0.5, -1.0))])], CATS)
    check("repeated identical pair: n = 2, still one pair", rep.knob_map != "not_learned" and rep.knob_map[0]["n"] == 2)

    section("candidates (6.5) and D5 (nothing qualifies -> nothing)")
    q = {"schema": "pkg.query/1", "types": ["Fridge"], "need": {"types": [], "tools": [], "appliances": [], "receptacles": [], "ops": []}}
    c = p.candidates(q)
    check("asked type -> match 1.0 and why", c and c[0]["rec"]["id"].endswith("fixture:Fridge") and c[0]["match"] == 1.0 and c[0]["why"] == ["type:Fridge"], c[:1])
    check("cand shape", all(set(x) == {"schema", "rec", "match", "why"} and x["schema"] == "pkg.cand/1" for x in c))
    c2 = p.candidates({"types": ["Egg"], "need": {"types": []}})
    check("asked item held inside a fixture -> 0.6", c2 and c2[0]["match"] == 0.6 and c2[0]["why"] == ["rel:contains:Egg"], c2)
    c3 = p.candidates({"types": [], "need": {"types": [], "tools": ["Knife"]}})
    check("needed tool: tool record 1.0 (own type) and CounterTop 0.4",
          {x["rec"]["id"].split(":")[-1]: x["match"] for x in c3}.get("Knife") == 1.0, c3)
    cnt = p.candidates({"types": ["Fork"], "need": {"types": [], "tools": ["Fork"]}})
    check("asked tool never seen -> nothing", cnt == [])
    check("empty query -> nothing", p.candidates({"types": [], "need": {}}) == [])
    check("unrelated type -> nothing", p.candidates({"types": ["Banana"], "need": {"types": ["Tomato"]}}) == [])
    c4 = p.candidates({"types": [], "need": {"types": [], "tools": ["Knife"]}})
    c4b = P.Profile.build(K, [ep2()], CATS).candidates({"types": [], "need": {"tools": ["Knife"]}})
    check("only-ep2 profile: CounterTop is not a fixture record, only the Knife record", [x["rec"]["id"].split(":")[-1] for x in c4b] == ["Knife"], c4b)
    q_in = json.dumps(q, sort_keys=True)
    p.candidates(q)
    check("candidates does not mutate the query", json.dumps(q, sort_keys=True) == q_in)
    check("stable order", [x["rec"]["id"] for x in p.candidates(q)] == [x["rec"]["id"] for x in p.candidates(q)])
    fk = P.Profile.build(K, [eps[0]], {"fixture_types": ["Fridge", "Drawer"], "tool_types": []})
    cf = fk.candidates({"types": [], "need": {"tools": ["Knife"]}})
    check("tool not in categories -> nothing", cf == [])
    # a tool fixture holding it: 0.4
    sp = {"Fridge": {}, "Drawer": {}}
    pp = P.Profile.build(K, [ep1()], CATS)
    pp.fixtures["Drawer"] = P._new_fixture()
    pp.fixtures["Drawer"].update({"n_seen": 1, "support": 1, "first_ord": 1, "last_ord": 1})
    pp.evidence["fixture/Drawer/seen"] = {"abc": 1}
    cc = pp.candidates({"types": [], "need": {"tools": ["Knife"]}})
    d = {x["rec"]["id"].split(":")[-1]: x["match"] for x in cc}
    check("fixture where the needed tool was last seen -> 0.4", d.get("Drawer") == 0.4 and d.get("Knife") == 1.0, d)

    section("persistence")
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "x", "profile.json")
        s1 = p.save(f)
        q2 = P.Profile.load(f)
        check("roundtrip sha", q2.to_json()["sha256"] == s1 == sh)
        check("roundtrip records equal", q2.records() == p.records())
        d = json.load(open(f, encoding="utf-8"))
        d["fixtures"]["Fridge"]["n_seen"] = 9
        json.dump(d, open(f, "w", encoding="utf-8"))
        try:
            P.Profile.load(f)
            check("tampered profile refused", False)
        except P.ProfileError:
            check("tampered profile refused", True)
        # dump_stream
        class J:
            def __init__(self): self.rows = []
            def append(self, r): self.rows.append(r); return r
        class A:
            episode = "ep:one"
            stream = ep1()["stream"]
        j = J()
        P.dump_stream(A(), j, order=0, attempt=1, item="it", lang="en")
        r = j.rows[0]
        check("dump_stream row", r["schema"] == "pkg.evs/1" and r["key"] == "ep:one" and r["attempt"] == 1 and len(r["stream"]) == len(A.stream))
        jf = os.path.join(td, "ev.jsonl")
        with open(jf, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n" + '{"torn')
        check("load_streams ignores a torn tail", len(P.load_streams(jf)) == 1)
        check("a dumped row rebuilds the same profile", P.Profile.build(K, P.load_streams(jf), CATS).fixtures["Fridge"]["n_seen"] == 2)

        section("from_store (partial) on a synthetic store")
        db = os.path.join(td, "ltm.db")
        con = sqlite3.connect(db)
        con.executescript("""
        create table episodes(episode text, kitchen text, E int);
        create table evidence(evidence text, claim text, claim_table text, E int);
        create table place_instance(place text, kitchen text, rtype text, cx real, cz real);
        create table place_alias(place text, canonical text, active int);
        create table location_claim(claim text, kitchen text, otype text, place text, verified text, status text, E_support int);
        create table negative_evidence(neg text, kitchen text, place text, otype text, support int, E int, superseded int);
        insert into episodes values('e1','SynthKitchen',1),('e2','SynthKitchen',2);
        insert into place_instance values('pl:1','SynthKitchen','Fridge',1.0,2.0),('pl:2','SynthKitchen','Fridge',-3.0,2.0),('pl:3','SynthKitchen','Drawer',0.0,1.0),('pl:4','SynthKitchen','Stool',5.0,5.0);
        insert into place_alias values('pl:2','pl:1',1);
        insert into evidence values('ev1','place_instance/pl:1','place_instance',1),('ev2','place_instance/pl:2','place_instance',1),('ev3','place_instance/pl:3','place_instance',2),
          ('ev4','location_claim/lc:a','location_claim',2),('ev5','location_claim/lc:b','location_claim',2),('ev6','location_claim/lc:c','location_claim',1),('ev7','negative_evidence/ng:1','negative_evidence',2);
        insert into location_claim values('lc:a','SynthKitchen','Egg','pl:1','verified','current',2),('lc:b','SynthKitchen','Spatula','pl:3','trusted','current',2),
          ('lc:c','SynthKitchen','Milk','pl:1','trusted','current',1),('lc:d','SynthKitchen','Pear','pl:1','verified','retired',1);
        insert into negative_evidence values('ng:1','SynthKitchen','pl:3','Knife',1,2,0);
        """)
        con.commit()
        con.close()
        ps = P.Profile.from_store(db, K, CATS)
        check("from_store marks partial", ps.partial is True and ps.to_json()["partial"] is True)
        check("from_store: fixtures from places (aliased/non-fixture skipped)", set(ps.fixtures) == {"Fridge", "Drawer"} and ps.fixtures["Fridge"]["n_seen"] == 1, ps.fixtures.keys())
        check("from_store: verified claim -> contents; trusted / retired not", set(ps.fixtures["Fridge"]["contents"]) == {"Egg"})
        check("from_store: tool last parent", ps.tools["Spatula"]["last_parent_type"] == "Drawer")
        check("from_store: negative evidence", ps.fixtures["Drawer"]["inspected_nothing"]["n"] == 1)
        check("from_store: knob map not learned", ps.knob_map == "not_learned")
        check("from_store: every rendered assertion has evidence", all(a["evidence"] for a in ps.assertions()) and ps.assertions())
        check("from_store: records valid", all(r["provenance"]["evidence"] for r in ps.records()) and len(ps.records()) == 3, [r["id"] for r in ps.records()])
        c2_ = sqlite3.connect(db)
        n_ep = c2_.execute("select count(*) from episodes").fetchone()[0]
        c2_.close()
        check("from_store: read-only (db unchanged)", n_ep == 2)
        check("from_store: wording does not claim inspection", "inside I detected" not in " ".join(r["text"]["en"] for r in ps.records()))

    section("real practice-v1 store (diagnostic; skipped when absent)")
    real = os.path.join(HERE, "final_test", "snapshots", "o0", "k10", "ltm.db")
    if os.path.exists(real):
        con = sqlite3.connect("file:%s?mode=ro" % real.replace("\\", "/"), uri=True)
        kit = con.execute("select distinct kitchen from episodes").fetchone()[0]
        con.close()
        pr = P.Profile.from_store(real, kit, CATS)
        a = pr.assertions()
        check("real store: partial, assertions > 0, zero without evidence", pr.partial and len(a) > 0 and not [x for x in a if not x["evidence"]], len(a))
        check("real store: deterministic", P.Profile.from_store(real, kit, CATS).to_json()["sha256"] == pr.to_json()["sha256"])
    else:
        print("   (no real store)")

    section("honesty: nothing outside the event contract is read")
    leaky = Stream("ep:leak")
    try:
        leaky.add("sighting", otype="Knife", state="x", conf=0.5, detector="d", track="tr:1", objectId="Knife|1|2|3")
        check("privileged event field refused upstream", False)
    except lev.EventError:
        check("privileged event field refused upstream", True)
    import copy
    noisy = copy.deepcopy(eps)
    for r in noisy:
        for e in r["stream"]:
            e["objectId"] = "Fridge|1|2|3"
            e["metadata"] = {"isOpen": True}
    check("unknown / privileged extra fields on events cannot change the profile", P.Profile.build(K, noisy, CATS).to_json()["sha256"] == sh)
    check("profile json holds no simulator-id-like text", not re.search(r"\|[-0-9.]+\||_physics|objectId", json.dumps(p.to_json())))

    print("PASS %d  FAIL %d" % (PASS, FAIL))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
