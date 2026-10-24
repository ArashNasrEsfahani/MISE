"""Gate for K-round6 (stage K-A2): profile pkg-1.r6 = pkg-1.r4 + need_rule (goal_schema.need_trigger -> homeowner inventory +
frozen uses table + one instruction line, word -> type affordance hints removed) + type_scope (D2: the Knife is filtered from every
agent-facing text). Offline (Windows python); no simulator, no LLM. Gate sentences are general kitchen language written for the gate,
except the r4 regression block, which rebuilds the frozen round-4 rows of the original / fresh sets and compares bytes."""
import hashlib, inspect, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "final_test", "signal"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ltm_package as LP, goal_schema as GS, type_scope as TS
from signal_pkg_plan import build_builder, LeaveOut, KITCHEN
from signal_plan import load, holding_of
N = [0, 0]


def check(name, ok, info=""):
    N[0 if ok else 1] += 1
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(info)) if (info and not ok) else ""))


stores, counter, _p, cats, _n = build_builder(set())


def pkg(text, lang="en", profile=LP.PROFILE_R6, held=None, off=(), item=None, summary=()):
    q = LP.make_query_profile(profile, text, lang, held, list(summary))
    st = dict(stores)
    if item:
        for c in ("template", "procedure", "profile", "places", "example"):
            if c in st: st[c] = LeaveOut(st[c], item)
    prm = LP.profile_params(profile, off=off) if off else LP.profile_params(profile)
    return LP.PackageBuilder(KITCHEN, st, counter, prm, categories=cats).build(q)


def T(text, lang=None):
    return GS.need_trigger(text, lang)


def bundle(p):
    t = p["text"]
    i = t.find("Kitchen inventory: the homeowner reports")
    return t[i:] if i >= 0 else ""


def outside(p):
    t = p["text"]
    i = t.find("Kitchen inventory: the homeowner reports")
    return t[:i] if i >= 0 else t


USES = json.load(open(LP.USES_PATH, encoding="utf-8"))
INV = json.load(open(LP.INVENTORY_PATH, encoding="utf-8"))
VOCAB = json.load(open(os.path.join(os.path.dirname(HERE), "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]

# ---- P: profile and switches
p6 = LP.profile_params(LP.PROFILE_R6)
pr = p6["precision"]
check("P1 r6 is a known profile with every r6 mechanism on and the D2 filter on", LP.PROFILE_R6 in LP.PROFILES
      and all(pr.get(m) for m in LP.R6_MECHANISMS) and pr.get("type_scope"))
check("P2 r6 = r4 mechanisms + need_rule", LP.R6_MECHANISMS == LP.R4_MECHANISMS + ("need_rule",))
pof = LP.profile_params(LP.PROFILE_R6, off=("need_rule",))
check("P3 off=need_rule switches only the need rule and changes the params sha", (not pof["precision"]["need_rule"]) and pof["precision"]["request_ops"]
      and pof["precision"]["type_scope"] and pof["sha256"] != p6["sha256"])
try:
    LP.profile_params(LP.PROFILE_R4, off=("need_rule",)); bad = False
except ValueError:
    bad = True
check("P4 need_rule cannot be switched for r4", bad)
try:
    LP.profile_params(LP.PROFILE_R6, off=("type_scope",)); bad = False
except ValueError:
    bad = True
check("P5 the D2 filter is not a switchable mechanism", bad)
check("P6 r4 params carry no r6 keys", not ({"need_rule", "type_scope", "uses_sha"} & set(LP.profile_params(LP.PROFILE_R4)["precision"])))
check("P7 the params pin the uses table and inventory sha", pr["uses_sha"] == hashlib.sha256(open(LP.USES_PATH, "rb").read()).hexdigest()
      and pr["inventory_sha"] == hashlib.sha256(open(LP.INVENTORY_PATH, "rb").read()).hexdigest())
check("P8 the trigger reads only the text, the language and its own parse", list(inspect.signature(GS.need_trigger).parameters) == ["text", "lang", "schema"])

# ---- T: the need trigger, one meaning in EN and FA
fires = [("I need something to stir the soup with.", "placeholder"), ("I need something to wipe the table with.", "placeholder"),
         ("Could use... I need something to open this jar with.", "placeholder"), ("I want something to drink.", "placeholder"),
         ("I need somewhere to put my keys.", "placeholder"), ("I'd like something to eat.", "placeholder"),
         ("چیزی برای هم زدن سوپ لازم دارم.", "placeholder"), ("یه چیزی می‌خوام که باهاش میز را پاک کنم.", "placeholder"),
         ("جایی لازم دارم که کلیدهایم را بگذارم.", "placeholder"),
         ("I want this tomato warm.", "desired_state"), ("I'd like the mug clean.", "desired_state"),
         ("I want to eat this egg hot tomorrow.", "desired_state"), ("گوجه را گرم می‌خواهم.", "desired_state")]
for s, kind in fires:
    r = T(s)
    check("T1 fires (%s): %s" % (kind, s), r["fires"] and r["kind"] == kind, r)
quiet = [("I don't need anything to stir with.", "negated"), ("I need nothing to stir with.", "negated"),
         ("Do I need something to stir with?", "question"), ("Is the tomato warm?", None), ("I like warm tomatoes.", "preference_only"),
         ("I prefer my milk cold.", "preference_only"), ("The tomato is warm.", None), ("Cold.", None), ("warm", None), ("سرد", None), ("گرم", None),
         ("I need a spoon to stir the soup.", None), ("Put the mug in the sink, I want it clean.", None),
         ("Bring me something to stir with.", None), ("I want you to put the cup in the cabinet.", None),
         ("چیزی برای هم زدن لازم ندارم.", "negated"), ("آیا چیزی برای هم زدن لازم دارم؟", "question"), ("دوست دارم گوجه گرم باشد.", "preference_only"),
         ("نمی‌خواهم گوجه سرد باشد.", "negated"), ("گوجه سرد است.", None), ("Pick up the spoon.", None), ("قاشق را بردار.", None),
         ("I want to drink this milk cold tomorrow.", None)]   # milk is not a known type: no NAMED object
for s, why in quiet:
    r = T(s)
    check("T2 does not fire (%s): %s" % (why or "no need", s), (not r["fires"]) and (why is None or r["reason"] == why), r)
pairs = [("I need something to stir the soup with.", "چیزی برای هم زدن سوپ لازم دارم."), ("I want this tomato warm.", "گوجه را گرم می‌خواهم."),
         ("I don't need anything to stir with.", "چیزی برای هم زدن لازم ندارم.")]
for e, f in pairs:
    a, b = T(e, "en"), T(f, "fa")
    check("T3 EN and FA give the same verdict: %s" % e, (a["fires"], a["kind"]) == (b["fires"], b["kind"]), (a, b))
r = T("I need something to stir the soup with.")
check("T4 the trigger names no type and no operation target", not any(v in json.dumps(r) for v in VOCAB))
lexraw = open(GS.NEED_LEX_PATH, encoding="utf-8").read()
check("T5 the cue lexicon has no type name and no item id", not re.search(r"\bg-l[123]-", lexraw) and not any(re.search(r"\b%s\b" % re.escape(t), lexraw) for t in VOCAB))
check("T6 the trigger is deterministic", T("I need something to stir with.") == T("I need something to stir with."))

# ---- U: the uses table
rows = USES["rows"]
scope = TS.filter_types(VOCAB)
check("U1 one row per in-scope FloorPlan1 type, Knife absent", sorted(r["type"] for r in rows) == sorted(scope) and "Knife" not in [r["type"] for r in rows]
      and len(rows) == len(VOCAB) - 1)
check("U2 rows are in a fixed alphabetical order", [r["type"] for r in rows] == sorted(r["type"] for r in rows))
check("U3 one template: a short use phrase plus at most one more", all(r["use"].count(";") <= 1 and len(r["use"].split()) <= 10 and r["use"][0].islower() for r in rows))
check("U4 no row names an action or tool of the planner contract", not any(re.search(r"\b(pickup|put_in|pick_and_place|toggle_on|toggle_off|slice|search_for)\b", r["use"]) for r in rows))
check("U5 the ButterKnife row grants no simulator capability (no SliceObject / slice action)", "SliceObject" not in json.dumps(USES)
      and USES["capability_rule"].startswith("everyday use only"))
check("U6 load_uses_table keeps the file order and filters D2", [t for t, _u in LP.load_uses_table()] == [r["type"] for r in rows])

# ---- I: the homeowner inventory (option C)
inv = LP.load_inventory(KITCHEN)
check("I1 the inventory is labelled, privileged and D2-filtered", INV["privileged"] and "homeowner" in INV["label"] and "Knife" not in inv and len(inv) == len(scope))
check("I2 an unknown kitchen has no inventory", LP.load_inventory("FloorPlan999") == [])

# ---- N: a NEED package
a = pkg("I need something to stir the soup with.")
b = pkg("I need something to read while I wait.")
f = pkg("چیزی برای هم زدن سوپ لازم دارم.", "fa")
check("N1 a NEED presentation is active and carries the bundle", a["need_rule"]["active"] and a["need_rule"]["bundle"] and "Kitchen inventory:" in a["text"])
check("N2 every permitted row is shown, in the fixed order (no matching, no ranking, no truncation)",
      [l.split(":")[0] for l in bundle(a).splitlines()[1:-2]] == [r["type"] for r in rows] and len(bundle(a).splitlines()) == len(rows) + 3)
check("N3 the bundle is the same for different needs (no matching to the need word)", bundle(a) == bundle(b))
check("N4 the bundle is the same for EN and FA", bundle(a) == bundle(f))
check("N5 the instruction line is Astra's wording, last line", bundle(a).splitlines()[-1] == LP.NEED_INSTRUCTION
      == "Select suitable in-scope resources and plan to satisfy the stated need or desired state, using supported actions.")
check("N6 the reported and the seen lists are separate; nothing seen yet", "reported, not seen by you" in bundle(a) and "Seen by you so far: none yet." in bundle(a))
s = pkg("I need something to stir the soup with.", summary=["Apple [CounterTop]", "Knife [Drawer]", "Bowl x2 [Cabinet]"])
check("N7 the seen list comes from the observation summary only, D2-filtered", "Seen by you so far: Apple, Bowl." in s["text"])
check("N8 no 'A tool that fits' line and no affordance type outside the bundle", "A tool that fits" not in a["text"] and a["need_audit"] == []
      and not re.search(r"\b(Spoon|Ladle)\b", outside(a)))
c = pkg("I want to eat this bread hot later.")
check("N9 a desired state keeps its meaning: no heat step, no appliance outside the bundle", c["need_rule"]["active"] and c["need_audit"] == []
      and not re.search(r"\b(heat|Microwave|StoveBurner|StoveKnob)\b", outside(c)) and "(stated: hot)" in c["text"])
check("N10 the retrieved part stays within the 300 budget; the bundle is outside it", a["tokens"]["retrieved_max"] <= 300 and a["tokens"]["need_bundle"] > 300
      and a["tokens"]["max"] >= a["tokens"]["need_bundle"])
prov = {x["id"]: x for x in a["provenance"]}
check("N11 provenance: the inventory is a privileged, authorized briefing record; table and line are attributed",
      prov.get("briefing/%s/homeowner-inventory" % KITCHEN, {}).get("privileged") is True and prov.get("procedure/*/uses-table", {}).get("source", "").startswith("table:")
      and prov.get("procedure/*/need-instruction", {}).get("source") == "amendment:R6-D1" and LP.audit_provenance(a) == [])
check("N12 the bundle text has no Persian character", not any("؀" <= ch <= "ۿ" for ch in f["text"]))
n0 = pkg("I need something to stir the soup with.", off=("need_rule",))
n4 = pkg("I need something to stir the soup with.", profile=LP.PROFILE_R4)
check("N13 need_rule off (STRICT) gives the r4 package text exactly", n0["text"] == n4["text"] and not n0["need_rule"]["active"])
q = LP.strip_need_hints(LP.make_query_profile(LP.PROFILE_R6, "I want to eat this bread hot later.", "en", None, []))
check("N14 strip_need_hints drops implied operations and affordance types, keeps stated ones", "heat" not in q["ops"] and "eat" in q["ops"]
      and q["intent"]["affordance_types"] == [] and not any(k.get("via_state") for k in q["intent"]["key"]))

# ---- R: a non-NEED request is the r4 package (plus the D2 filter only)
for s_, l_ in (("Put the egg in the microwave, close the door, and turn it on.", "en"), ("Pick up the spoon.", "en"), ("The egg is rotten.", "en"),
               ("کاسه را داخل یکی از کابینت‌ها بگذار و درش را ببند.", "fa"), ("Put the apple in the fridge and close the door.", "en")):
    x6, x4 = pkg(s_, l_), pkg(s_, l_, profile=LP.PROFILE_R4)
    check("R1 non-NEED = r4: %s" % s_, x6["text"] == x4["text"] and not x6["need_rule"]["active"], (x6["text"], x4["text"]))

# ---- D: the D2 type filter
k = pkg("Pick up the knife.")
check("D1 a request naming the Knife renders no Knife text", not TS.mentions_excluded(k["text"]))
check("D2 ButterKnife is not the Knife", TS.mentions_excluded("pickup ButterKnife") == [] and TS.mentions_excluded("pickup Knife") == ["Knife"])
check("D3 the filter list is exactly the Knife", TS.EXCLUDED_TYPES == ("Knife",) and TS.filter_types(["Knife", "ButterKnife"]) == ["ButterKnife"])

# ---- F: the frozen round-4 rows rebuild byte-identically under r4 (r6 changed nothing in r4)
man, by = load()
_g = json.load(open(os.path.join(HERE, "dataset", "general.json"), encoding="utf-8"))
by = dict(by, **{a["id"]: a["item"] for a in _g.get("archived", [])})      # A26: the frozen r4 rows include archived items
#: A26 (K-B2b) successor version: the only r4 rows that may change, each explained (both conditions, measured 2026-10-13 with the
#: real counter): the egg FA text (decision 8) and the two archived Knife items (their request line loses the Knife / the tool
#: template is no longer left out for an item that left the dataset). Every other row must still rebuild byte-identically.
A26_EXPLAINED = {("g-l2-microwave-run-egg", "fa"), ("g-l3-cutlery-sink", "en"), ("g-l3-cutlery-sink", "fa"),
                 ("g-l3-tool-cut", "en"), ("g-l3-tool-cut", "fa")}
for cond in ("universal", "strict"):
    MECH4 = json.load(open(os.path.join(HERE, "final_test", "signal", "r4", "mechanisms.json"), encoding="utf-8"))
    rows4 = [json.loads(l) for l in open(os.path.join(HERE, "final_test", "signal", "r4", "pkg_rows_pkg_1_r4_%s.jsonl" % cond), encoding="utf-8")]
    st0, cnt0, _x, cats0, _y = build_builder({r["item"] for r in rows4})
    same, chg = 0, set()
    for r in rows4:
        it = by[r["item"]]
        off = tuple(r.get("off") or ())
        prm = LP.profile_params(LP.PROFILE_R4, off=off)
        qq = LP.make_query_profile(LP.PROFILE_R4, it[r["lang"]], r["lang"], holding_of(it), [])
        st = dict(st0)
        for c_ in ("template", "procedure", "profile", "places", "example"):
            if c_ in st: st[c_] = LeaveOut(st[c_], r["item"])
        _eq = LP.PackageBuilder(KITCHEN, st, cnt0, prm, categories=cats0).build(qq)["text"] == r["text"]
        same += _eq
        if not _eq:
            chg.add((r["item"], r["lang"]))
    # A26 replaces: "F1 r4 %s rows rebuild byte-identically" (same == len(rows4))
    check("F1 r4 %s rows rebuild byte-identically except the A26-explained ones (%d of %d; changed %s)" % (cond, same, len(rows4), sorted(chg)),
          chg <= A26_EXPLAINED and same == len(rows4) - len(chg))

print("pkg_round6 selftest: %d/%d" % (N[0], N[0] + N[1]))
sys.exit(0 if not N[1] else 1)
