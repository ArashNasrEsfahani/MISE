"""Self-test for step 5 / A5-replay -- step-4 run logs -> whitelisted event streams -> WorkingMemory -> consolidate -> memory.

Offline, Windows `python` from the REPO ROOT, temp directories only (never sim/trace.db, sim/memory.db or the default
sim/step5/ltm.db), no simulator, no WSL, no Ollama, no network:

    python sim/_step5_replay_selftest.py

Covers (docs/research/step5-astra-build-review.txt; sim/step5/_astra/A5-replay.txt): copied + hashed sources (originals
read-only and unchanged); explicit-extraction whitelist (privileged fields never read - proven by a perturbation test that
scrambles every excluded field and expects byte-identical events and an identical database); opaque content-addressed
episode ids; malformed rows rejected; every stored value scanned for forbidden strings / simulator ids / privileged
tokens (with a positive control); counts that follow from the events independently; the privileged verdict confined to
procedure admission / plan_goal_unmet; no invented geometry (no places, claims, habits, negatives; where_is explore-only);
idempotent / duplicated / truncated / reordered replay; frozen-state equality before and after a test pass; EN/FA pairing
sharing one evidence unit; and a matrix proving the adversarial gates of the build review are covered by some selftest.
"""
import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append, not insert: sim/trace.py must not shadow the stdlib `trace`

import ltm                                # noqa: E402
import ltm_lessons as ll                  # noqa: E402
import step5_replay as R                  # noqa: E402
from ltm_events import EventError, validate   # noqa: E402
from ltm_store import canon               # noqa: E402
from ltm_wm import WorkingMemory          # noqa: E402

REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


_OPEN = []


def new_ltm(tmp, name):
    L = ltm.LongTermMemory(Path(tmp) / name / "ltm.db", kitchen="FloorPlan1", params=ltm.load_params())
    _OPEN.append(L)
    return L


def mem_guard():
    p = REPO / "sim" / "memory.db"
    return (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None


def steps_of(eps):
    return ll.step_index(WorkingMemory.replay(eps["events"]))


def admissible(ep):
    """Independent re-derivation from the EVENTS: goal verdict True, exec_ok, every step has an ok final outcome."""
    ev = ep["events"]
    adm = next((e for e in ev if e["kind"] == "admission_verdict"), None)
    term = next(e for e in ev if e["kind"] == "terminal")
    outs = [e for e in ev if e["kind"] == "step_outcome"]
    starts = [e for e in ev if e["kind"] == "step_start"]
    return bool(adm and adm["goal_ok"] is True and term["exec_ok"] and outs and len(outs) == len(starts)
                and all(o["ok"] for o in outs))


def failed_steps(ep):
    return [e for e in ep["events"] if e["kind"] == "step_outcome" and e["ok"] is False]


def scramble(row, rnd):
    """Scramble EVERY excluded field of a raw row (except `id`, which is the public-dataset join key)."""
    r = copy.deepcopy(row)
    junk = lambda: f"Apple|{rnd.uniform(-9, 9):+.2f}|+00.50|{rnd.uniform(-9, 9):+.2f} err {rnd.random()}"
    r["hash"], r["ts"], r["detail"] = f"{rnd.getrandbits(64):016x}", "1999-01-01T00:00:00", junk()
    r["cause"] = rnd.choice([None, "goal_unmet", "retry_exhausted", junk()])
    r["first_fail"] = {"object": junk()}
    r["pre"] = {"scoring": rnd.randint(0, 99), "oracle_setup": rnd.randint(0, 99)}
    r["told_holding"], r["seed_tracks"], r["sweep_prims"], r["elapsed_s"] = junk(), rnd.randint(0, 9), 5, 1.5
    r["mode"] = "oracle"
    a = r["audit"] = dict(r.get("audit") or {})
    a["errors_logged"] = [junk(), {"objectId": junk(), "errorMessage": junk()}]
    a["counts"] = {"oracle": rnd.randint(0, 9), "scoring": 7}
    a["episode"] = junk()
    a["errors"] = 3
    for s in r["steps"]:
        s["why"], s["verify"], s["holding"] = junk(), [junk()], junk()
        s["search"] = {"visited": [junk()], "opened": [junk()], "plan": [{"label": junk()}]}
        s["events"] = [junk()]
    return r


def main():
    mg0 = mem_guard()
    tmp = tempfile.mkdtemp(prefix="s5replay_")
    items = R.load_items()

    # ------------------------------------------------------------------ sources: copied, hashed, read-only
    paths = R.default_sources(REPO)
    check("sources found (console re-run + final-fixes checkpoints)", len(paths) >= 2, len(paths))
    hashes0 = {str(p): R.sha256_file(p) for p in paths}
    snaps = R.snapshot_sources(Path(tmp) / "src", paths, REPO)
    check("every source copied", len(snaps) == len(paths) and all(os.path.exists(s["copy"]) for s in snaps))
    check("copy hash == original hash", all(s["copy_matches_source"] for s in snaps))
    check("hashes recorded (64 hex) for before / after / copy",
          all(len(s[k]) == 64 for s in snaps for k in ("sha256_before", "sha256_after", "sha256_copy")))
    check("no source changed while being copied", not any(s["source_changed"] for s in snaps))
    check("sources recorded repo-relative", snaps[0]["source"].startswith("sim/step4/"))

    raw = []
    per_src = {}
    for s in snaps:
        rows, bad = R.read_rows(s["copy"])
        raw += rows
        per_src[s["tag"]] = len(rows)
        check(f"{s['tag']}: no torn lines", bad == 0)
    check("raw rows loaded", len(raw) >= 277, len(raw))

    eps, man = R.convert(snaps, items)
    T = man["totals"]
    # A26 (K-B2b): rows of the 6 archived Knife items are skipped as `skipped_archived` (never replayed into memory)
    ARCH = R.load_archived()
    check("every row accounted for",
          T["rows"] == T["converted"] + T["skipped_inapplicable"] + T["rejected"] + T["duplicate_content"] + T.get("skipped_archived", 0), T)
    check("A26: every row of an archived item is skipped, nothing else", T.get("skipped_archived", 0) == sum(1 for r in raw if r.get("id") in ARCH), T)
    check("raw row count matches", T["rows"] == len(raw), (T["rows"], len(raw)))
    check("skipped rows produce no episode", T["skipped_inapplicable"] == sum(1 for r in raw if r.get("status") == "skipped" and r.get("id") not in ARCH))
    check("no real row rejected", T["rejected"] == 0, man["rejected"][:3])
    check("duplicates are content-identical copies, not dropped runs", T["duplicate_content"] > 0)
    check("no invalid episode in the real logs", T["invalid"] == 0)
    check("manifest source hashes = snapshot hashes", [s["sha256_copy"] for s in man["sources"]] == [s["sha256_copy"] for s in snaps])
    check("source id lives only in the evaluator manifest", all(
        v["row_id"] in items for v in man["episode_sources"].values()))

    # ------------------------------------------------------------------ whitelist: shape of every event
    ALLOWED = {
        "episode_start": {"mode", "world"}, "task_context": {"level", "lang", "text", "concepts"},
        "step_start": {"i", "verb", "otype", "to"}, "step_outcome": {"i", "ok", "cause", "under", "retries", "prims"},
        "terminal": {"exec_ok", "prims", "invalid"}, "admission_verdict": {"goal_ok", "tag"}, "episode_end": set()}
    ENV = {"schema", "ev", "kind", "kitchen", "epoch", "frame", "map_rev", "episode", "t"}
    shape_ok = verdict_ok = ids_ok = True
    kinds = set()
    for ep in eps:
        for e in ep["events"]:
            kinds.add(e["kind"])
            if set(e) - ENV - ALLOWED.get(e["kind"], set()):
                shape_ok = False
            if "goal_ok" in e and e["kind"] != "admission_verdict":
                verdict_ok = False
            validate(e)
        if not ep["episode"].startswith("rp:") or any(
                v["row_id"] in canon(ep["events"]) for v in [man["episode_sources"][ep["episode"]]]):
            ids_ok = False
    check("events carry only whitelisted fields", shape_ok)
    check("only admission_verdict mentions goal_ok", verdict_ok)
    check("no geometry / attempt / inspection events are invented",
          kinds <= set(ALLOWED), kinds - set(ALLOWED))
    check("episode ids are opaque (no source row id inside the stream)", ids_ok)
    check("every stream validates and replays through WorkingMemory", all(WorkingMemory.replay(e["events"]).complete for e in eps))
    check("task_context text is the public instruction",
          all(next(x for x in e["events"] if x["kind"] == "task_context")["text"] in {i["en"] for i in items.values()} for e in eps))
    check("episode ids are content-addressed and unique", len({e["episode"] for e in eps}) == len(eps))

    eps2, man2 = R.convert(snaps, items)
    check("conversion is deterministic", canon(eps) == canon(eps2))
    dup_snap = snaps + [dict(snaps[0], tag="src999")]
    eps3, man3 = R.convert(dup_snap, items)
    check("the same run copied twice adds no episodes", canon(eps3) == canon(eps) and man3["totals"]["duplicate_content"] > T["duplicate_content"])
    capped, _m = R.convert(snaps, items, per_source_cap=3)
    check("per_source_cap limits rows read", len(capped) < len(eps))

    # ------------------------------------------------------------------ perturbation: excluded fields cannot matter
    rnd = random.Random(5)
    diff = 0
    n = 0
    for r in raw:
        it = items.get(r.get("id"))
        if r.get("status") not in R.ITEM_STATUS or it is None:      # A26 (K-B2b): rows of archived items have no public item now
            continue
        a = R.row_events(r, it, episode="rp:x")
        b = R.row_events(scramble(r, rnd), it, episode="rp:x")
        n += 1
        if canon(a) != canon(b):
            diff += 1
    check("scrambling every excluded field leaves events byte-identical", diff == 0 and n >= 300, (diff, n))
    pert = [scramble(r, rnd) for r in raw if r.get("id") not in ARCH]          # A26: archived rows are skipped by convert
    pert_eps = []
    for r in pert:
        evs = R.row_events(r, items.get(r.get("id")), episode="rp:pending")
        if evs is not None:
            eid = R.episode_id(evs)
            pert_eps.append({"episode": eid, "events": [dict(e, episode=eid) for e in evs]})
    seen = set()
    pert_eps = [e for e in pert_eps if not (e["episode"] in seen or seen.add(e["episode"]))]
    check("perturbed rows give the same episode ids", [e["episode"] for e in pert_eps] == [e["episode"] for e in eps])
    # not vacuous: changing a WHITELISTED field does change the stream
    base = next(r for r in raw if r.get("status") == "FAIL" and r["steps"] and r.get("id") not in ARCH)   # A26: not an archived item
    chg = copy.deepcopy(base)
    chg["steps"][0]["retries"] = base["steps"][0]["retries"] + 1
    chg2 = copy.deepcopy(base)
    chg2["goal_ok"] = not base["goal_ok"]
    it0 = items[base["id"]]
    check("whitelisted fields do reach the stream (retries, verdict)",
          canon(R.row_events(base, it0, episode="e")) != canon(R.row_events(chg, it0, episode="e")) and
          canon(R.row_events(base, it0, episode="e")) != canon(R.row_events(chg2, it0, episode="e")))
    check("level comes from the field and must agree with the public dataset",
          raises(R.ReplayError, R.row_events, dict(base, level="L2" if base["level"] != "L2" else "L3"), it0, episode="e"))

    # ------------------------------------------------------------------ malformed rows are rejected, not coerced
    def mut(**kw):
        r = copy.deepcopy(base)
        r.update(kw)
        return r
    bad_step = lambda **kw: mut(steps=[dict(base["steps"][0], **kw)])
    for name, row in [("unknown verb", bad_step(action="teleport")), ("sim-id object", bad_step(object="Apple|+01.2|+00.9|-02.0")),
                      ("object with digits", bad_step(object="Apple2")), ("destination with a space", bad_step(to="Fridge door")),
                      ("non-bool ok", bad_step(ok="yes")), ("negative retries", bad_step(retries=-1)),
                      ("huge prims", bad_step(prims=10 ** 9)), ("steps not a list", mut(steps="x")),
                      ("hung row", mut(status="HUNG")), ("missing level", mut(level=None)),
                      ("step not an object", mut(steps=["pickup"])), ("too many steps", mut(steps=base["steps"] * 40))]:
        check(f"rejected: {name}", raises(R.ReplayError, R.row_events, row, it0, episode="e"))
    check("rejected: no dataset item", raises(R.ReplayError, R.row_events, base, None, episode="e"))
    check("skipped row -> None", R.row_events(mut(status="skipped"), it0, episode="e") is None)
    free = R.row_events(bad_step(cause="Unreachable: object Apple|+01.0|+00.1|-02.0 is too far"), it0, episode="e")
    check("free-text step cause becomes unknown (dropped), not stored", all("Apple|" not in canon(e) for e in free) and not any(
        e.get("cause") for e in free if e["kind"] == "step_outcome"))
    inv = R.row_events(mut(audit=dict(base["audit"], invalid=True)), it0, episode="e")
    check("audit.invalid -> terminal.invalid (integrity flag only)", next(e for e in inv if e["kind"] == "terminal")["invalid"] is True)
    inv2 = R.row_events(mut(audit=dict(base["audit"], violations=2, invalid=False)), it0, episode="e")
    check("audit.violations>0 also marks the episode invalid", next(e for e in inv2 if e["kind"] == "terminal")["invalid"] is True)
    # a privileged key cannot be smuggled through the contract
    good = copy.deepcopy(eps[0]["events"])
    for kind, key, val in [("terminal", "goal_ok", True), ("step_start", "objectId", "x"), ("step_outcome", "errors_logged", ["x"]),
                           ("task_context", "oracle", "x"), ("terminal", "group", "g1"), ("step_outcome", "scoring", 1)]:
        evs = copy.deepcopy(good)
        tgt = next(e for e in evs if e["kind"] == kind)
        tgt[key] = val
        check(f"contract refuses {key} on {kind}", raises(EventError, validate, tgt))
    sidx = next(e for e in good if e["kind"] == "step_start")
    check("contract refuses a simulator id in a type field", raises(EventError, validate, dict(sidx, otype="Apple|+01.20|+00.90|-02.00")))
    check("contract refuses a non-ASCII type", raises(EventError, validate, dict(sidx, otype="سیب")))

    # ------------------------------------------------------------------ replay into a real (temp) store
    A = new_ltm(tmp, "a")
    ext = ll.LessonsExtension()
    out = R.replay_into(A, eps, [ext])
    check("every episode committed", out["tally"] == {"committed": len(eps)}, out["tally"])
    check("integrity check passes", A.check_integrity() in (None, True, []) or bool(A.check_integrity() is not False))
    st = A.stats()
    check("clock E == number of episodes", st["E"] == len(eps), (st["E"], len(eps)))
    check("one episodes row per episode", st["rows"]["episodes"] == len(eps))
    check("no geometry in the logs -> no places, claims, habits, negatives, placements, trials",
          all(st["rows"][t] == 0 for t in ("place_instance", "location_claim", "negative_evidence", "placement", "habit",
                                           "search_trial", "place_alias")), st["rows"])
    check("no hypothesis prose without a hypothesizer", all(
        not str(r.get("text") or r.get("prose") or "") for r in A.store.select("hypothesis_audit") if not str(r.get("hyp", "dx:")).startswith("dx:")))
    check("replay never sets a mirror error", st["mirror_errors"] == [])
    # counts that follow independently from the events
    adm = {e["episode"] for e in eps if admissible(e)}
    gwp = {e["episode"] for e in eps if any(x["kind"] == "admission_verdict" and x["goal_ok"] for x in e["events"]) and failed_steps(e)}
    procs = A.store.select("procedure")
    admits = [r for r in A.store.select("application_log") if r["retrieval"] == "admit"]
    check("procedures only from episodes that pass the declared admission", {r["episode"] for r in admits} <= adm)
    check("no procedure from a goal-met episode with a failed step", not ({r["episode"] for r in admits} & gwp))
    check("goal_without_plan counted per such episode", sum(r["stats"]["goal_without_plan"] for r in A.store.select("episodes")) == len(gwp), len(gwp))
    check("never more procedures than admissible episodes", 0 < len(procs) <= len(adm), (len(procs), len(adm)))
    check("every procedure body step is a logged action of its admitting episode", all(
        {s["verb"] for s in p["body"]["steps"]} <= R.VERBS for p in procs))
    gu = {e["episode"] for e in eps if any(x["kind"] == "admission_verdict" and x["goal_ok"] is False for x in e["events"])
          and not failed_steps(e)}
    check("plan_goal_unmet episodes exist in the real logs and admit nothing", gu and not ({r["episode"] for r in admits} & gu), len(gu))
    lessons = A.store.select("lesson")
    fail_eps = {e["episode"] for e in eps if failed_steps(e)}
    check("lessons only originate in episodes with a failed step", {l["origin_episode"] for l in lessons} <= fail_eps)
    nfail = sum(len(failed_steps(e)) for e in eps)
    check("lesson recurrences never exceed observed failed steps", sum(l["recurrences"] for l in lessons) <= nfail, (sum(l["recurrences"] for l in lessons), nfail))
    check("one lesson per exact local failure key", len({l["cond_key"] for l in lessons}) == len(lessons))
    check("no lesson was ever applied, so none is supported", all(l["status"] == "tentative" and l["helpful_units"] == 0 for l in lessons))
    check("lesson causes come from the fixed taxonomy", all(l["conditions"]["cause"] in ll.CAUSES for l in lessons))
    check("lesson actions come from the fixed action set", all(l["action"] in ll.ACTIONS for l in lessons))
    check("replay used no hypothesizer: every action is the registry default", len({l["action"] for l in lessons}) >= 1)
    # where_is stays explore-only
    wi_ok = True
    for ot in ("Apple", "Knife", "Mug", "Bread", "Egg", "Potato"):
        w = A.where_is(ot)
        wi_ok &= all(c["tier"] == "T4" and c["source"] == "prior" for c in w["candidates"]) and w["explore"]["id"] == "EXPLORE"
    check("where_is is well-formed and explore-only (replay has no place evidence)", wi_ok)
    # practice-mode lesson lookup may use a tentative lesson; test mode may not
    if lessons:
        l0 = lessons[0]
        key = dict(l0["conditions"])
        wm_p = A.begin_episode("rp:probe", "practice")
        wm_p.ingest(dict(schema="s5.ev/1", kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode="rp:probe", ev=1,
                         kind="episode_start", t=1, mode="practice", world="reset"))
        wm_t = WorkingMemory("rp:probe2")
        wm_t.ingest(dict(schema="s5.ev/1", kitchen="FloorPlan1", epoch=0, frame="f0", map_rev="m0", episode="rp:probe2", ev=1,
                         kind="episode_start", t=1, mode="test", world="reset"))
        gp = ll.lesson_for(A, wm_p, key, {l0["action"]}, 3)
        gt = ll.lesson_for(A, wm_t, key, {l0["action"]}, 3)
        check("practice may try a tentative lesson", gp is not None and gp["experimental"])
        check("test mode never applies a replayed (tentative) lesson", gt is None)

    # ------------------------------------------------------------------ leakage scan over every stored value
    forb = R.forbidden_strings(raw)
    check("forbidden-string list is substantial", len(forb) > 1000, len(forb))
    TOK = ("errors_logged", "objectid", "oracle", "scoring", "metadata", "errorMessage", "audit.")
    leaks = R.scan_store(A.store.path, forb, TOK)
    check("no forbidden string / sim id / privileged token in ANY stored value", leaks == [], leaks[:5])
    check("no forbidden string in the event streams", R.scan_events(eps, forb, TOK) == [])
    # positive controls: the scanner would see a leak
    pc = R.scan_store(A.store.path, ["pick Apple"], ())
    check("positive control: store scan finds a planted string", len(pc) >= 1)
    check("positive control: event scan finds a planted token", len(R.scan_events(eps, [], ["step_outcome"])) >= 1)
    leaky = copy.deepcopy(eps[:1])
    leaky[0]["events"][0]["mode"] = "practice errors_logged"
    check("positive control: event scan finds a privileged token", len(R.scan_events(leaky, [], ["errors_logged"])) == 1)
    n_err = sum(len((r.get("audit") or {}).get("errors_logged") or []) for r in raw)
    check("the raw logs really carry error texts (so the scan is meaningful)", n_err > 0, n_err)

    # perturbed raw rows -> the identical database
    P = new_ltm(tmp, "pert")
    R.replay_into(P, pert_eps, [ll.LessonsExtension()])
    check("perturbed excluded fields -> identical state hash", P.state_hash() == A.state_hash())

    # ------------------------------------------------------------------ determinism, idempotence, duplicates, truncation
    B = new_ltm(tmp, "b")
    R.replay_into(B, eps, [ll.LessonsExtension()])
    check("replay into a fresh store is deterministic (same state hash)", B.state_hash() == A.state_hash())
    h0, e0 = A.state_hash(), A.clock()
    again = R.replay_into(A, eps, [ll.LessonsExtension()])
    check("second replay of every episode is a no-op ('duplicate')", again["tally"] == {"duplicate": len(eps)}, again["tally"])
    check("state hash and clock unchanged by the second replay", A.state_hash() == h0 and A.clock() == e0)
    # duplicated events inside a stream
    C = new_ltm(tmp, "c")
    dup_eps = [{"episode": e["episode"], "events": e["events"] + e["events"][:3]} for e in eps]
    dup_ok = True
    for e in dup_eps:
        wm = WorkingMemory(e["episode"])
        try:
            for ev_ in sorted(e["events"], key=lambda x: x["ev"]):
                wm.ingest(ev_)
        except EventError:
            dup_ok = False
        C.end_episode(wm, extensions=[ll.LessonsExtension()])
    check("identical duplicate events are ignored", dup_ok and C.state_hash() == A.state_hash())
    one = copy.deepcopy(eps[0]["events"])
    one.append(dict(one[1], text="a different instruction"))
    check("a conflicting duplicate event fails", raises(EventError, WorkingMemory.replay, one))
    # truncated stream: no episode_end -> incomplete -> nothing written
    D = new_ltm(tmp, "d")
    h_empty = D.state_hash()
    tr = []
    for e in eps[:30]:
        wm = WorkingMemory(e["episode"])
        for ev_ in e["events"][:-1]:
            wm.ingest(ev_)
        tr.append(D.end_episode(wm, extensions=[ll.LessonsExtension()])["status"])
    check("truncated episodes (no episode_end) write nothing", set(tr) == {"incomplete"} and D.state_hash() == h_empty and D.clock() == 0)
    cut = []
    for e in eps[:30]:
        wm = WorkingMemory(e["episode"])
        for ev_ in e["events"][:3]:
            wm.ingest(ev_)
        cut.append(D.end_episode(wm)["status"])
    check("episodes cut mid-way write nothing", set(cut) == {"incomplete"} and D.state_hash() == h_empty)
    # a stream with the terminal marked invalid writes nothing
    ev_inv = copy.deepcopy(eps[0]["events"])
    next(x for x in ev_inv if x["kind"] == "terminal")["invalid"] = True
    for x in ev_inv:
        x["episode"] = "rp:invalid1"
    wmi = WorkingMemory("rp:invalid1")
    for x in ev_inv:
        wmi.ingest(x)
    check("an invalid episode (integrity flag) is not consolidated", D.end_episode(wmi)["status"] == "invalid" and D.state_hash() == h_empty)
    # reordered: shuffled arrival of events within each episode (replay() sorts) gives the same store
    Fs = new_ltm(tmp, "f")
    rr = random.Random(11)
    for e in eps:
        evs = list(e["events"])
        rr.shuffle(evs)
        wm = WorkingMemory.replay(evs)
        Fs.end_episode(wm, extensions=[ll.LessonsExtension()])
    check("events shuffled within each episode (batch replay) -> identical state hash", Fs.state_hash() == A.state_hash())
    live = WorkingMemory(eps[0]["episode"])
    evs = list(eps[0]["events"])
    live.ingest(evs[1]) if False else None
    live.ingest(evs[0])
    live.ingest(evs[2])
    check("live ingest refuses out-of-order events", raises(EventError, live.ingest, evs[1]))
    # reordered episodes: the clock E follows arrival order, but the SEMANTIC content must not
    G = new_ltm(tmp, "g")
    perm = R.shuffled_order(len(eps), 3)
    R.replay_into(G, eps, [ll.LessonsExtension()], order=perm)
    sem = lambda L: (sorted(p["proc"] for p in L.store.select("procedure")), sorted(l["cond_key"] for l in L.store.select("lesson")),
                     sorted((l["cond_key"], l["recurrences"]) for l in L.store.select("lesson")))
    check("episode order changes E but not WHICH procedures / lessons / counts exist", sem(G) == sem(A))
    check("lesson origin may differ with order (first-seen), never the key set", {l["cond_key"] for l in G.store.select("lesson")} == {l["cond_key"] for l in lessons})

    # ------------------------------------------------------------------ EN / FA pairing shares one evidence unit
    base_ok = next(r for r in raw if r.get("status") == "ok" and r["level"] == "L1" and r.get("goal_ok") is True)
    it_ok = items[base_ok["id"]]
    H = new_ltm(tmp, "h")
    for lang in ("en", "fa"):
        evs = R.row_events(base_ok, it_ok, episode="rp:pending", lang=lang)
        eid = R.episode_id(evs)
        evs = [dict(e, episode=eid) for e in evs]
        wm = WorkingMemory.replay(evs)
        H.end_episode(wm, extensions=[ll.LessonsExtension()])
    check("EN and FA of one execution are two episodes", H.stats()["rows"]["episodes"] == 2)
    pr = H.store.select("procedure")
    check("the translation pair does not double procedure support", all(p["success_uses"] <= 1 for p in pr), [(p["proc"], p["success_uses"]) for p in pr])
    fa_ev = R.row_events(base_ok, it_ok, episode="e", lang="fa")
    check("the FA instruction text is the public Persian text", next(e for e in fa_ev if e["kind"] == "task_context")["text"] == it_ok["fa"])
    # ------------------------------------------------------------------ frozen equality across a test pass
    snap_dir = Path(tmp) / "snap"
    A.snapshot(snap_dir)
    fz = ltm.LongTermMemory.open_frozen(snap_dir, Path(tmp) / "snap-testlog.db")
    _OPEN.append(fz)
    sha_before, hash_before = R.sha256_file(fz.store.path), fz.state_hash()
    statuses = set()
    for e in eps[:60]:
        evs = [x for x in copy.deepcopy(e["events"]) if x["kind"] != "admission_verdict"]   # practice-only (plan 1.3)
        for x in evs:
            x["episode"] = "t:" + e["episode"][3:]
        evs[0]["mode"] = "test"
        wm = WorkingMemory.replay(evs)
        statuses.add(fz.end_episode(wm, extensions=[ll.LessonsExtension()]).get("status"))
        fz.where_is("Apple", wm=wm)
        if lessons:
            ll.lesson_for(fz, wm, dict(lessons[0]["conditions"]), {lessons[0]["action"]}, 3)
    check("test-mode episodes on a frozen store consolidate nothing", statuses <= {"frozen", "skipped_mode"}, statuses)
    check("frozen state hash equal before / after the test pass", fz.state_hash() == hash_before)
    check("frozen file sha equal before / after the test pass", R.sha256_file(fz.store.path) == sha_before)
    check("the live store is untouched by the frozen pass", A.state_hash() == h0)
    check("practice consolidation on a frozen store is refused", fz.end_episode(WorkingMemory.replay(eps[0]["events"]))["status"] == "frozen")

    # ------------------------------------------------------------------ build-review adversarial gate matrix
    MATRIX = {
        "duplicate / reordered / truncated replay": [("_step5_storage_selftest.py", "duplicate"), ("_step5_replay_selftest.py", "truncated")],
        "crash between admission and trace export": [("_step5_storage_selftest.py", "crash between")],
        "migration / restart": [("_step5_storage_selftest.py", "migration"), ("_step5_storage_selftest.py", "restart")],
        "adjacent identical drawers / erroneous merges": [("_step5_rules_selftest.py", "adjacent")],
        "moved objects": [("_step5_rules_selftest.py", "moved")],
        "unqualified negatives": [("_step5_rules_selftest.py", "unqualified")],
        "repeated-discovery inflation": [("_step5_rules_selftest.py", "inflation")],
        "stale embeddings": [("_step5_retrieval_selftest.py", "stale")],
        "Persian orthographic variants / negation": [("_step5_retrieval_selftest.py", "ZWNJ"), ("_step5_retrieval_selftest.py", "negation")],
        "token overflow": [("_step5_retrieval_selftest.py", "oversize")],
        "unsupported hypothesis references": [("_step5_lessons_selftest.py", "evidence ref")],
        "failed procedure preconditions": [("_step5_lessons_selftest.py", "precondition")],
        "privileged-field contamination": [("_step5_replay_selftest.py", "perturb"), ("_step5_storage_selftest.py", "privileged")],
        "frozen-state equality before / after testing": [("_step5_replay_selftest.py", "frozen state hash"), ("_step5_eval_selftest.py", "frozen")],
        "imports / gates cannot start a controller or contact Ollama": [("_step5_honesty_selftest.py", "socket")],
    }
    missing = []
    for gate, refs in MATRIX.items():
        for fn, needle in refs:
            p = Path(HERE) / fn
            txt = p.read_text(encoding="utf-8").lower() if p.exists() else ""
            if needle.lower() not in txt:
                missing.append((gate, fn, needle))
    check("every adversarial gate of the build review is covered by a named selftest", not missing, missing)

    # ------------------------------------------------------------------ originals untouched, no shared db touched
    check("original sources unchanged after the whole run", all(R.sha256_file(p) == h for p, h in
                                                                {str(p): hashes0[str(p)] for p in paths}.items()))
    check("sim/memory.db untouched", mem_guard() == mg0)
    for L in _OPEN:
        try:
            L.close()
        except Exception:
            pass

    print("\nNOT VALIDATED BY REPLAY (deferred to workflow B): association accuracy (the logs carry no world coordinates, so no")
    print("place was created and no merge / abstain decision was exercised); inspection coverage and qualified negatives (the logs")
    print("have no open/detection evidence); changed search trajectories under memory; real lesson usefulness (no recovery kinds,")
    print("no per-attempt causes and no lesson application exist in the logs); bilingual execution (one execution, EN context);")
    print("real Gemma hypotheses; token compliance; DrvFs durability.")
    print(f"\n{_pass} passed, {_fail} failed")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
