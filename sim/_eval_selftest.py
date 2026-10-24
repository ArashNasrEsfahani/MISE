"""Pure Phase-8 gate for the evaluation harness (sim/eval.py).

No AI2-THOR, no Ollama, no network, no live stores. Runs on a bare Windows box with the
stock `python` interpreter and asserts eval.py's PURE metric/aggregation layer against
hand-computed cases, then (when present) validates the real dataset against the schema.

What it checks:
  1. eval.py imports with ai2thor POISONED out of sys.modules -- proving it does NOT pull in
     thor_actions / ai2thor at module top (that would be a bug; the sim deps must be lazy).
  2. levenshtein            -- known edit distances (empty, sub, del, ins, compound).
  3. action_label / labels  -- the exact "<norm_action> <canon_type>" convention + nav stripping.
  4. pc_score  (THESIS Eq.2) -- both-empty=1.0, identical=1.0, one-edit, disjoint, prefix-half.
  5. tds_score (THESIS Eqs.3-5) -- perfect=1.0, partial P/R, empty-gen / empty-req / disjoint
                               edge cases (all zeros, NO ZeroDivisionError).
  6. sr_score               -- equality + contains assertions, multi-assert AND, empty-goal False.
  7. aggregate()            -- SR/PC/TDS rolled up per level / lang / model over a synthetic set.
  8. dataset (if present)   -- every record validates via levels.CommandRecord and every gold
                               action label uses a known action name from skills.skills_manifest().

This gate is store-free: the metric, aggregation and dataset-read paths are pure, so nothing
needs a temporary sqlite/JSON redirect. Run it:  python sim/_eval_selftest.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ------------------------------------------------------------------------------------------------
# Tiny PASS/FAIL/SKIP recorder + a per-check table.
# ------------------------------------------------------------------------------------------------
_RESULTS = []  # list of (name, status)  status in {"PASS","FAIL","SKIP"}


def _record(name, status, note=""):
    _RESULTS.append((name, status))
    line = "  [%-4s] %s" % (status, name)
    if note:
        line += "  (%s)" % note
    print(line)


def check(name, cond):
    _record(name, "PASS" if cond else "FAIL")
    return bool(cond)


def skip(name, note):
    _record(name, "SKIP", note)


def approx(a, b, tol=1e-9):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


# ------------------------------------------------------------------------------------------------
# CHECK 1 -- import eval.py with ai2thor absent (proves the sim deps are lazy).
# ------------------------------------------------------------------------------------------------
def import_eval_without_ai2thor():
    """Poison ai2thor so ANY attempt to import it (directly or via thor_actions) raises, then import
    eval. If eval imported thor_actions/ai2thor at module top this fails -> reported as a BUG."""
    for name in ("ai2thor", "ai2thor.controller", "ai2thor.platform"):
        sys.modules[name] = None            # None sentinel -> `import ai2thor` raises ImportError
    ev = None
    try:
        import eval as ev  # noqa: A004  (local module sim/eval.py, not the builtin)
        check("eval.py imports with ai2thor poisoned (sim deps are lazy)", True)
    except Exception as exc:
        check("eval.py imports with ai2thor poisoned (sim deps are lazy) -- BUG: %s" % exc, False)
        return None
    # Belt-and-suspenders: importing eval must NOT have dragged in the heavy sim module either.
    check("thor_actions NOT imported at eval module top", "thor_actions" not in sys.modules)
    check("eval exposes the pure metric layer",
          all(hasattr(ev, n) for n in ("levenshtein", "action_label", "labels", "sr_score",
                                       "pc_score", "tds_score", "aggregate", "load_dataset",
                                       "norm_action", "canon_type", "repair_plan", "pickup_target")))
    return ev


# ------------------------------------------------------------------------------------------------
# CHECK 2 -- levenshtein over label sequences.
# ------------------------------------------------------------------------------------------------
def test_levenshtein(ev):
    lev = ev.levenshtein
    check("lev empty/empty = 0", lev([], []) == 0)
    check("lev ['a']/[] = 1 (deletion)", lev(["a"], []) == 1)
    check("lev []/['a','b','c'] = 3 (insertions)", lev([], ["a", "b", "c"]) == 3)
    check("lev one substitution = 1", lev(["a", "b", "c"], ["a", "x", "c"]) == 1)
    check("lev one deletion = 1", lev(["a", "b"], ["b"]) == 1)
    check("lev one insertion = 1", lev(["a", "b", "c"], ["a", "b", "c", "d"]) == 1)
    check("lev compound (del+ins) = 2", lev(["a", "b", "c", "d"], ["b", "c", "d", "e"]) == 2)
    check("lev whole-label elements = 1",
          lev(["pickup Apple", "open Fridge"], ["open Fridge"]) == 1)


# ------------------------------------------------------------------------------------------------
# CHECK 3 -- action_label / labels convention.
# ------------------------------------------------------------------------------------------------
def test_action_label(ev):
    al = ev.action_label
    check("label 'pickup Apple'", al({"action": "pickup", "object": "Apple"}) == "pickup Apple")
    check("label 'open Fridge'", al({"action": "open", "object": "Fridge"}) == "open Fridge")
    check("label object wins over to",
          al({"action": "pick_and_place", "object": "Apple", "to": "Fridge"}) == "pick_and_place Apple")
    check("label to-only fallback", al({"action": "put_in", "to": "Fridge"}) == "put_in Fridge")
    check("label alias heat->heat_object", al({"action": "heat", "object": "Potato"}) == "heat_object Potato")
    check("label alias turn_on_all->switch_all",
          al({"action": "turn_on_all", "object": "LightSwitch"}) == "switch_all LightSwitch")
    check("label alias pick_up->pickup", al({"action": "pick_up", "object": "Egg"}) == "pickup Egg")
    check("label canon type refrigerator->Fridge",
          al({"action": "open", "object": "refrigerator"}) == "open Fridge")
    check("label canon type sink->SinkBasin",
          al({"action": "clean", "object": "sink"}) == "clean SinkBasin")
    # labels() = S_gen: action_label per step, dropping navigation/perception/no-op helpers.
    plan = [{"action": "go_to", "object": "Knife"}, {"action": "pickup", "object": "Knife"},
            {"action": "scan_for", "object": "Tomato"}, {"action": "slice", "object": "Tomato"},
            {"action": "done"}]
    check("labels() strips nav/perception/no-op helpers",
          ev.labels(plan) == ["pickup Knife", "slice Tomato"])


# ------------------------------------------------------------------------------------------------
# CHECK 4 -- pc_score (THESIS Eq.2).
# ------------------------------------------------------------------------------------------------
def test_pc(ev):
    pc = ev.pc_score
    check("pc both empty = 1.0", pc([], []) == 1.0)
    check("pc identical = 1.0", pc(["pickup Apple"], ["pickup Apple"]) == 1.0)
    check("pc one edit (1 - 1/3) = 2/3", approx(pc(["a", "b", "c"], ["a", "x", "c"]), 2.0 / 3.0))
    check("pc disjoint same length = 0.0", pc(["a", "b"], ["x", "y"]) == 0.0)
    check("pc empty gen vs one gt = 0.0", pc([], ["pickup Apple"]) == 0.0)
    check("pc prefix half (1 - 2/4) = 0.5", approx(pc(["a", "b", "c", "d"], ["a", "b"]), 0.5))


# ------------------------------------------------------------------------------------------------
# CHECK 5 -- tds_score (THESIS Eqs.3-5).
# ------------------------------------------------------------------------------------------------
def test_tds(ev):
    tds = ev.tds_score
    full = tds({"pickup Apple", "cool_object Apple"}, {"pickup Apple", "cool_object Apple"})
    check("tds perfect P=R=F1=1.0",
          approx(full["precision"], 1.0) and approx(full["recall"], 1.0) and approx(full["f1"], 1.0))
    part = tds({"a", "b", "c"}, {"a", "b"})  # inter=2 -> P=2/3, R=1.0, F1=0.8
    check("tds partial P=2/3", approx(part["precision"], 2.0 / 3.0))
    check("tds partial R=1.0", approx(part["recall"], 1.0))
    check("tds partial F1=0.8", approx(part["f1"], 0.8))
    check("tds empty gen -> zeros", tds(set(), {"a"}) == {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    check("tds empty req -> zeros", tds({"a"}, set()) == {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    check("tds both empty -> zeros", tds(set(), set()) == {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    check("tds disjoint F1=0 (no ZeroDivision)", tds({"a"}, {"b"})["f1"] == 0.0)


# ------------------------------------------------------------------------------------------------
# CHECK 6 -- sr_score.
# ------------------------------------------------------------------------------------------------
def test_sr(ev):
    sr = ev.sr_score
    state = {
        ("Fridge", "isOpen"): True,
        ("Apple", "parentReceptacles"): ["Fridge|+00.1|+00.9|-01.3"],
        ("LightSwitch", "isToggled"): False,
    }
    getf = lambda t, k: state.get((t, k))
    check("sr equality holds -> True",
          sr([{"type": "Fridge", "key": "isOpen", "value": True}], getf) is True)
    check("sr equality mismatch -> False",
          sr([{"type": "Fridge", "key": "isOpen", "value": False}], getf) is False)
    check("sr contains matches -> True",
          sr([{"type": "Apple", "key": "parentReceptacles", "contains": "Fridge"}], getf) is True)
    check("sr contains misses -> False",
          sr([{"type": "Apple", "key": "parentReceptacles", "contains": "Microwave"}], getf) is False)
    check("sr multi-assert AND -> True",
          sr([{"type": "Fridge", "key": "isOpen", "value": True},
              {"type": "LightSwitch", "key": "isToggled", "value": False}], getf) is True)
    check("sr multi-assert one fails -> False",
          sr([{"type": "Fridge", "key": "isOpen", "value": True},
              {"type": "LightSwitch", "key": "isToggled", "value": True}], getf) is False)
    check("sr empty goal -> False", sr([], getf) is False)
    check("sr None goal -> False", sr(None, getf) is False)
    check("sr contains over None value -> False (no crash)",
          sr([{"type": "Nope", "key": "parentReceptacles", "contains": "Fridge"}], getf) is False)


# ------------------------------------------------------------------------------------------------
# CHECK 7 -- aggregate() on a synthetic per-record set (hand-computed rollups).
# ------------------------------------------------------------------------------------------------
def test_aggregate(ev):
    stamp = {"rag": "off", "loop": "off", "perception": "oracle", "scene": "FloorPlan1"}

    def rec(level, lang, model, sr, pc=None, tds_f1=None):
        r = {"level": level, "lang": lang, "model": model, "SR": sr,
             "PC": pc, "TDS": ({"precision": tds_f1, "recall": tds_f1, "f1": tds_f1}
                               if tds_f1 is not None else None)}
        r.update(stamp)
        return r

    fake = [
        rec("L1", "en", "A", True),
        rec("L1", "fa", "A", False),
        rec("L2", "en", "A", True, pc=0.5),
        rec("L2", "fa", "B", False, pc=1.0),
        rec("L3", "en", "B", True, tds_f1=0.8),
        rec("L3", "fa", "B", False, tds_f1=0.4),
    ]
    agg = ev.aggregate(fake)

    ov = agg["overall"]
    check("agg overall n=6", ov["n"] == 6)
    check("agg overall SR=0.5", approx(ov["SR"], 0.5))
    check("agg overall PC=0.75 (L2 only)", approx(ov["PC"], 0.75))
    check("agg overall TDS=0.6 (L3 only)", approx(ov["TDS"], 0.6))

    lv = agg["by_level"]
    check("agg L1 SR=0.5, PC/TDS None",
          approx(lv["L1"]["SR"], 0.5) and lv["L1"]["PC"] is None and lv["L1"]["TDS"] is None)
    check("agg L2 PC=0.75, TDS None", approx(lv["L2"]["PC"], 0.75) and lv["L2"]["TDS"] is None)
    check("agg L3 TDS=0.6, PC None", approx(lv["L3"]["TDS"], 0.6) and lv["L3"]["PC"] is None)

    lg = agg["by_lang"]
    check("agg lang en SR=1.0/PC=0.5/TDS=0.8",
          approx(lg["en"]["SR"], 1.0) and approx(lg["en"]["PC"], 0.5) and approx(lg["en"]["TDS"], 0.8))
    check("agg lang fa SR=0.0/PC=1.0/TDS=0.4",
          approx(lg["fa"]["SR"], 0.0) and approx(lg["fa"]["PC"], 1.0) and approx(lg["fa"]["TDS"], 0.4))

    md = agg["by_model"]
    check("agg model A SR=2/3, PC=0.5, TDS None",
          approx(md["A"]["SR"], 2.0 / 3.0) and approx(md["A"]["PC"], 0.5) and md["A"]["TDS"] is None)
    check("agg model B SR=1/3, PC=1.0, TDS=0.6",
          approx(md["B"]["SR"], 1.0 / 3.0) and approx(md["B"]["PC"], 1.0) and approx(md["B"]["TDS"], 0.6))

    key = "A|rag=off|loop=off|perc=oracle|FloorPlan1"
    check("agg by_config key present + n=3", key in agg["by_config"] and agg["by_config"][key]["n"] == 3)


# ------------------------------------------------------------------------------------------------
# CHECK 8 -- real dataset (if present) validates against the schema + action vocabulary + the
# gold-authoring contract (per-level sub-key, canonical nav-free labels) + END-TO-END coherence
# (the composite-preferring planner's intended plan must score PC/TDS == 1.0 against gold).
# ------------------------------------------------------------------------------------------------
# Canonical, real FloorPlan1 objectTypes gold labels may reference (CLAUDE.md safe list). Used to reject
# typo'd / non-canonical object tokens in gold (e.g. "refrigerator", "sink", "Aple").
_KNOWN_TYPES = {
    "Apple", "Bread", "Tomato", "Potato", "Lettuce", "Egg", "Mug", "Cup", "Bowl", "Plate", "Pan", "Pot",
    "Knife", "ButterKnife", "Fork", "Spoon", "Spatula", "Fridge", "Microwave", "Toaster", "CoffeeMachine",
    "StoveBurner", "StoveKnob", "Cabinet", "Drawer", "CounterTop", "Sink", "SinkBasin", "Faucet", "LightSwitch",
    "GarbageCan", "Bottle", "SoapBottle", "DishSponge", "PepperShaker", "SaltShaker", "HousePlant", "Window",
    "Stool",
}


# Canonical manifest action -> the SURFACE verb the planner prompt tends to emit (eval.norm_action must map it
# back). Used to synthesize a realistic "intended plan" from any record's gold, so this check is dataset-agnostic.
_SURFACE_VERB = {"heat_object": "heat", "cool_object": "cool", "wash_object": "wash",
                 "cook_food": "cook", "break_object": "break"}


def _intended_plan_for(rec):
    """The plan the composite-preferring PLANNER PROMPT is designed to emit for a given L2/L3 record, derived from
    the record's OWN gold. Each gold label '<action> <Object>' becomes a leading go_to (a nav step eval.labels()
    must strip) plus the manipulation using the SURFACE verb (which eval.norm_action() must normalize back). So
    eval.labels(plan) must reproduce the gold labels in order -> PC/TDS == 1.0. Works for any dataset; catches the
    atomic-vs-composite granularity trap (gold authored at a granularity the planner does not emit fails here
    instead of silently depressing the headline metric)."""
    gold = rec.gold or {}
    labs = gold.get("sequence") if rec.level == "L2" else gold.get("actions")
    plan = []
    for lab in (labs or []):
        toks = str(lab).split()
        if len(toks) < 2:
            continue
        act, obj = toks[0], " ".join(toks[1:])
        plan.append({"action": "go_to", "object": obj})                      # nav step -> must be stripped
        plan.append({"action": _SURFACE_VERB.get(act, act), "object": obj})  # surface verb -> must normalize
    return plan


def test_dataset(ev):
    import levels
    import skills

    path = os.path.join(HERE, "dataset", "commands.jsonl")
    if not os.path.exists(path):
        for nm in ("dataset commands.jsonl validates", "dataset gold labels use known action names",
                   "dataset per-level gold sub-key present (L2 sequence / L3 actions)",
                   "dataset gold labels are '<action> <ObjectType>' (>=2 tokens)",
                   "dataset gold labels are navigation-free", "dataset gold object tokens are canonical types",
                   "dataset intended composite plan -> PC/TDS==1.0"):
            skip(nm, "file absent (separate deliverable)")
        return

    known_actions = {s["name"] for s in skills.skills_manifest()}
    try:
        records = ev.load_dataset(path, lang="all", limit=None)
    except Exception as exc:
        check("dataset commands.jsonl loads + every record validates -- ERROR: %s" % exc, False)
        skip("dataset gold labels use known action names", "load failed")
        return
    check("dataset commands.jsonl loads + every record validates (n=%d)" % len(records), len(records) > 0)

    bad_labels, bad_goal, bad_sub, bad_arity, bad_nav, bad_type = [], [], [], [], [], []
    for rec in records:
        gold = rec.gold or {}
        # per-level gold sub-key: L2 MUST carry an ordered `sequence` (scored by PC), L3 MUST carry an `actions`
        # set (scored by TDS). Absent it, eval scores PC against [] / TDS against the empty set -- a silent
        # nonsense number rather than an error (the coverage hole FINDING 2 flagged).
        if rec.level == "L2" and not gold.get("sequence"):
            bad_sub.append("%s: L2 without 'sequence'" % rec.id)
        if rec.level == "L3" and not gold.get("actions"):
            bad_sub.append("%s: L3 without 'actions'" % rec.id)
        # goal shape (used by sr_score): each assertion is a dict with type+key and value-or-contains.
        for a in (gold.get("goal") or []):
            if not (isinstance(a, dict) and a.get("type") and a.get("key")
                    and ("value" in a or "contains" in a)):
                bad_goal.append("%s:%r" % (rec.id, a))
        # sequence (L2) + actions (L3) labels must be a well-formed canonical "<action> <ObjectType>": exactly a
        # known manifest action, >=2 tokens (an object is present), navigation-free (labels() drops nav helpers
        # from the generated side, so a nav label in gold could never intersect), and a canonical real objectType.
        for lab in list(gold.get("sequence") or []) + list(gold.get("actions") or []):
            toks = str(lab).split()
            if len(toks) < 2:
                bad_arity.append("%s:%r" % (rec.id, lab))
                continue
            act, obj = toks[0], " ".join(toks[1:])
            if act not in known_actions:
                bad_labels.append("%s:%r" % (rec.id, lab))
            if ev.norm_action(act) in ev._NAV_HELPERS:
                bad_nav.append("%s:%r" % (rec.id, lab))
            if ev.canon_type(obj) != obj or obj not in _KNOWN_TYPES:
                bad_type.append("%s:%r" % (rec.id, lab))
    check("dataset goal assertions well-formed" + (" (bad: %s)" % bad_goal[:3] if bad_goal else ""),
          not bad_goal)
    check("dataset gold labels use known action names"
          + (" (bad: %s)" % bad_labels[:3] if bad_labels else ""), not bad_labels)
    check("dataset per-level gold sub-key present (L2 sequence / L3 actions)"
          + (" (bad: %s)" % bad_sub[:3] if bad_sub else ""), not bad_sub)
    check("dataset gold labels are '<action> <ObjectType>' (>=2 tokens)"
          + (" (bad: %s)" % bad_arity[:3] if bad_arity else ""), not bad_arity)
    check("dataset gold labels are navigation-free (no go_to/scan_for/...)"
          + (" (bad: %s)" % bad_nav[:3] if bad_nav else ""), not bad_nav)
    check("dataset gold object tokens are canonical, real FloorPlan1 types"
          + (" (bad: %s)" % bad_type[:3] if bad_type else ""), not bad_type)

    # END-TO-END COHERENCE: push each fixtured record's INTENDED (composite-preferring) plan through eval.labels()
    # -- action-name normalization + nav stripping -- and require PC (L2) / TDS-f1 (L3) == 1.0 against gold. This
    # is the check metric-functions-in-isolation cannot do: it fails if gold is authored at a granularity the
    # planner does not emit (the atomic-vs-composite trap), rather than silently depressing the headline metric.
    coh_bad, n_coh = [], 0
    for rec in records:
        if rec.level not in ("L2", "L3"):
            continue
        plan = _intended_plan_for(rec)                          # synthesized from the record's own gold
        gold = rec.gold or {}
        gen = ev.labels(plan)
        if rec.level == "L2":
            score = ev.pc_score(gen, gold.get("sequence") or [])
            tag = "PC"
        elif rec.level == "L3":
            score = ev.tds_score(set(gen), set(gold.get("actions") or []))["f1"]
            tag = "TDS"
        else:
            continue
        if approx(score, 1.0):
            n_coh += 1
        else:
            coh_bad.append("%s %s=%.3f" % (rec.id, tag, score))
    check("dataset intended composite plan -> PC/TDS==1.0 for every fixtured L2/L3 record (n=%d)" % n_coh
          + (" (bad: %s)" % coh_bad[:3] if coh_bad else ""), not coh_bad and n_coh > 0)


# ------------------------------------------------------------------------------------------------
# CHECK 9 -- store redirection: an ablation can run against snapshot copies of the research stores.
# ------------------------------------------------------------------------------------------------
def test_store_flags(ev):
    cfg = ev._config_from_args(ev._build_parser().parse_args([]))
    check("store flags default to empty (= the canonical sim/ stores, unchanged behaviour)",
          cfg.get("kb_path") == "" and cfg.get("memory_path") == "" and cfg.get("loop_reports_path") == "")
    cfg = ev._config_from_args(ev._build_parser().parse_args(
        ["--kb", "/tmp/x/kb.json", "--memory", "/tmp/x/m.db", "--loop-reports", "/tmp/x/r.jsonl"]))
    check("--kb / --memory / --loop-reports are echoed into the run config (the results doc records them)",
          (cfg["kb_path"], cfg["memory_path"], cfg["loop_reports_path"])
          == ("/tmp/x/kb.json", "/tmp/x/m.db", "/tmp/x/r.jsonl"))
    src = open(os.path.join(HERE, "eval.py"), encoding="utf-8").read()
    check("close_loop is given the run's loop-report path", 'report_path=(cfg.get("loop_reports_path") or None)' in src)
    check("webapp's search singletons are bound to the run's stores", "webapp._kb, webapp._memory = kb, mem" in src)
    cfg = ev._config_from_args(ev._build_parser().parse_args(["--weights-dir", "/tmp/cand"]))
    check("--weights-dir is echoed into the run config, and defaults to empty (= sim/weights)",
          cfg["weights_dir"] == "/tmp/cand" and ev._config_from_args(ev._build_parser().parse_args([]))["weights_dir"] == "")
    check("the perceived observation is resolved from the run's weights dir",
          'weights_dir=cfg.get("weights_dir") or None' in src)


# ------------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 78)
    print("PHASE-8 EVAL HARNESS SELF-TEST (pure: no ai2thor / Ollama / network / stores)")
    print("=" * 78)

    ev = import_eval_without_ai2thor()
    if ev is None:
        # Cannot proceed without the module under test.
        _summary_and_exit()

    test_levenshtein(ev)
    test_action_label(ev)
    test_pc(ev)
    test_tds(ev)
    test_sr(ev)
    test_aggregate(ev)
    test_dataset(ev)
    test_store_flags(ev)

    _summary_and_exit()


def _summary_and_exit():
    n_pass = sum(1 for _, s in _RESULTS if s == "PASS")
    n_fail = sum(1 for _, s in _RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s in _RESULTS if s == "SKIP")
    print("-" * 78)
    print("checks: %d total | %d PASS | %d FAIL | %d SKIP" % (len(_RESULTS), n_pass, n_fail, n_skip))
    if n_fail:
        failed = [name for name, s in _RESULTS if s == "FAIL"]
        print("FAILED: " + "; ".join(failed))
        print("=" * 78)
        sys.exit(1)
    print("EVAL SELFTEST OK")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
