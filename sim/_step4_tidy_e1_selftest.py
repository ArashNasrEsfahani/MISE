"""Offline selftest for P-round5 E1 (sim/honest_skills.py, 2026-10-13): the end-of-step tidy and a later toggle_on.
Windows `python`, no WSL/GPU/Ollama. Run from the repo root:  python sim/_step4_tidy_e1_selftest.py

E1: a later toggle_on of an openable appliance that switches on ONLY with its door closed (SWITCH_ON_CLOSED = Microwave)
does not count as "a later step needs it open", so the tidy closes a door this agent opened IMPLICITLY (put route,
composite) before that step. Unchanged: explicit opens are never tidied; any other later step that names the container
(put_in, close, pickup of the object found in it) still keeps it open; TIDY_SWITCH_CLOSED = False restores round 4.

1. unit: _needed_later on hand-made entries / upcoming steps (no simulator).
2. integration on the step-4 fake kitchen (sim/_step4_act_selftest.py's FakeThor, made to REFUSE ToggleObjectOn of an
   open Microwave as AI2-THOR does): pick_and_place Apple Microwave, toggle_on Microwave -> door closed by the tidy,
   toggle_on succeeds, the apple is inside and heated; with E1 off the round-4 failure is reproduced (retry_exhausted).
3. unchanged behaviours: put + close + toggle_on; explicit open + put_in + toggle_on (explicit open kept, toggle refused);
   heat composite + toggle_on; toggle_off later still keeps it open; a Fridge is not affected; no ids, 0 violations."""
import contextlib, io, json, os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TRACE_ENABLED", "0")
import honest_skills as hs  # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, info=""):
    N["pass" if cond else "fail"] += 1
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("" if cond else f"  -- {info}"))


print("1. unit: _needed_later")
sk = hs.HonestSkills.__new__(hs.HonestSkills)


def needed(entry, upcoming):
    sk.upcoming = upcoming
    return sk._needed_later(entry)


mw = {"type": "Microwave", "why": "put"}
check("constants: SWITCH_ON_CLOSED is the Microwave only; E1 on by default",
      hs.SWITCH_ON_CLOSED == frozenset({"Microwave"}) and hs.TIDY_SWITCH_CLOSED is True)
check("later toggle_on Microwave alone: NOT needed open (E1)", needed(mw, [{"action": "toggle_on", "object": "Microwave"}]) is False)
check("alias turn_on/switch_on resolved the same way when aliased",
      all(needed(mw, [{"action": a, "object": "Microwave"}]) is False for a in ("toggle_on",)
          + tuple(k for k, v in hs.ALIASES.items() if v == "toggle_on")))
check("later close Microwave: still needed (kept open for that step)", needed(mw, [{"action": "close", "object": "Microwave"}]) is True)
check("later put_in Microwave: still needed", needed(mw, [{"action": "put_in", "object": "Microwave"}]) is True)
check("later pick_and_place X to Microwave: still needed", needed(mw, [{"action": "pick_and_place", "object": "Egg", "to": "Microwave"}]) is True)
check("toggle_on then put_in later: still needed (any other naming step keeps it)",
      needed(mw, [{"action": "toggle_on", "object": "Microwave"}, {"action": "put_in", "object": "Microwave"}]) is True)
check("later toggle_off Microwave: unchanged (still counts as needed; E1 is toggle_on only)",
      needed(mw, [{"action": "toggle_off", "object": "Microwave"}]) is True)
check("a Fridge with a later toggle_on Fridge (not a switch-closed type): unchanged", needed({"type": "Fridge", "why": "put"},
      [{"action": "toggle_on", "object": "Fridge"}]) is True)
check("search-found target picked up later: still needed", needed({"type": "Microwave", "why": "search-found", "target": "Egg"},
      [{"action": "toggle_on", "object": "Microwave"}, {"action": "pickup", "object": "Egg"}]) is True)
hs.TIDY_SWITCH_CLOSED = False
check("E1 off: later toggle_on Microwave counts as needed again (round-4 behaviour)",
      needed(mw, [{"action": "toggle_on", "object": "Microwave"}]) is True)
hs.TIDY_SWITCH_CLOSED = True

print("2./3. integration on the step-4 fake kitchen")
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(os.path.join(HERE, "_step4_act_selftest.py"), run_name="_step4_act_for_e1")
setup, obj, EXH, FakeThor = G["setup"], G["obj"], G["EXH"], G["FakeThor"]
no_ids = G["no_ids"]
_orig = FakeThor.interact


def _strict_interact(self, action, kw):
    """AI2-THOR: a Microwave does not switch on while its door is open."""
    o = self.objs.get(kw.get("objectId")) if kw.get("objectId") else None
    if action == "ToggleObjectOn" and o is not None and o["type"] == "Microwave" and o.get("isOpen"):
        return False
    return _orig(self, action, kw)


FakeThor.interact = _strict_interact


def run(plan, e1=True):
    hs.TIDY_SWITCH_CLOSED = e1
    t, det, h, sk = setup(**EXH)
    out = []
    for i, st in enumerate(plan):
        out.append(sk.exec_step(st, upcoming=plan[i + 1:]))
    hs.TIDY_SWITCH_CLOSED = True
    return t, h, sk, out


P1 = [{"action": "pick_and_place", "object": "Apple", "to": "Microwave"}, {"action": "toggle_on", "object": "Microwave"}]
t, h, sk, r = run(P1)
check("E1 on: pick_and_place Apple Microwave then toggle_on -> both steps ok",
      r[0]["ok"] and r[1]["ok"], json.dumps([x["cause"] for x in r]))
check("...the tidy closed the door at the end of the put step (why=put, closed)",
      any(x["why"] == "put" and x["result"] == "closed" for x in r[0]["detail"].get("tidy") or []), json.dumps(r[0]["detail"].get("tidy")))
check("...apple inside the Microwave, Microwave on and closed, apple heated",
      obj(t, "Apple").get("inside") == obj(t, "Microwave")["id"] and obj(t, "Microwave")["isToggled"]
      and not obj(t, "Microwave")["isOpen"] and obj(t, "Apple").get("temperature") == "Hot")
check("...no ids in the results, 0 violations", all(no_ids(x) for x in r) and h.audit.summary()["violations"] == 0)
t, h, sk, r = run(P1, e1=False)
check("E1 off reproduces round 4: door kept open for the toggle_on, which is refused (retry_exhausted)",
      r[0]["ok"] and r[0]["detail"]["left_open"] == ["Microwave"] and not r[1]["ok"] and r[1]["cause"] == "retry_exhausted"
      and not obj(t, "Microwave")["isToggled"],
      json.dumps([x["cause"] for x in r]))
P2 = [{"action": "pick_and_place", "object": "Apple", "to": "Microwave"}, {"action": "close", "object": "Microwave"},
      {"action": "toggle_on", "object": "Microwave"}]
t, h, sk, r = run(P2)
check("put + close + toggle_on (the reference shape): unchanged, all ok, door left open for the close step",
      all(x["ok"] for x in r) and r[0]["detail"]["left_open"] == ["Microwave"] and obj(t, "Microwave")["isToggled"],
      json.dumps([(x["ok"], x["cause"]) for x in r]))
P3 = [{"action": "pickup", "object": "Apple"}, {"action": "open", "object": "Microwave"},
      {"action": "put_in", "object": "Microwave"}, {"action": "toggle_on", "object": "Microwave"}]
t, h, sk, r = run(P3)
check("explicit open + put_in + toggle_on: the explicit open is NOT tidied (contract), toggle_on refused as before",
      r[1]["ok"] and r[2]["ok"] and not r[3]["ok"] and obj(t, "Microwave")["isOpen"],
      json.dumps([(x["ok"], x["cause"]) for x in r]))
P4 = [{"action": "heat", "object": "Apple"}, {"action": "toggle_on", "object": "Microwave"}]
t, h, sk, r = run(P4)
check("heat composite then toggle_on: the door the route opened is closed by the tidy, toggle_on ok (apple in hand)",
      r[0]["ok"] and r[1]["ok"] and t.held and "Apple" in t.held and not obj(t, "Microwave")["isOpen"],
      json.dumps([(x["ok"], x["cause"]) for x in r]))
P5 = [{"action": "pick_and_place", "object": "Apple", "to": "Microwave"}, {"action": "toggle_off", "object": "Microwave"}]
t, h, sk, r = run(P5)
check("put then toggle_off later: unchanged (door kept open for that step)", r[0]["ok"] and r[0]["detail"]["left_open"] == ["Microwave"])
P6 = [{"action": "pick_and_place", "object": "Apple", "to": "Fridge"}, {"action": "toggle_on", "object": "Faucet"}]
t, h, sk, r = run(P6)
check("a Fridge put followed by an unrelated toggle_on: closed at step end as before",
      r[0]["ok"] and not obj(t, "Fridge")["isOpen"] and r[0]["detail"]["left_open"] == [])
FakeThor.interact = _orig
print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
