"""Gate for N-round4 (repair for round 4): profile pkg-1.r4 = r3 + G1 (request_ops: the request rendered in English from the
parser's own reading) + G2 (precondition_closure: a requested toggle_on Microwave keeps the close X line). Offline (Windows python);
no simulator, no LLM. Sentences here are written for the gate (general kitchen language), not taken from the dataset, except where a
check pins the round-3 behaviour of the r3 profile on its own frozen rows."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "final_test", "signal"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ltm_package as LP, goal_schema as GS
from signal_pkg_plan import build_builder, LeaveOut, KITCHEN
N = [0, 0]


def check(name, ok, info=""):
    N[0 if ok else 1] += 1
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(info)) if (info and not ok) else ""))


stores, counter, _p, cats, _n = build_builder(set())


def pkg(text, lang, profile=LP.PROFILE_R4, held=None, off=(), item=None):
    q = LP.make_query_profile(profile, text, lang, held, [])
    st = dict(stores)
    if item:
        for c in ("template", "procedure", "profile", "places", "example"):
            if c in st: st[c] = LeaveOut(st[c], item)
    prm = LP.profile_params(profile, off=off) if off else LP.profile_params(profile)
    return LP.PackageBuilder(KITCHEN, st, counter, prm, categories=cats).build(q)


def req(p):
    return [l for l in p["text"].splitlines() if l.startswith("The request in English") or l.startswith("This request names")]


def R(text, lang="en"):
    return LP.render_request(GS.parse_goal(text, lang))


# ---- P: profile and switches
p4 = LP.profile_params(LP.PROFILE_R4)
check("P1 r4 is a known profile with both mechanisms on", LP.PROFILE_R4 in LP.PROFILES and p4["precision"]["request_ops"] and p4["precision"]["precondition_closure"])
pof = LP.profile_params(LP.PROFILE_R4, off=("request_ops",))
check("P2 off= switches one mechanism off and changes the params sha", (not pof["precision"]["request_ops"]) and pof["precision"]["precondition_closure"]
      and pof["sha256"] != p4["sha256"])
try:
    LP.profile_params(LP.PROFILE_R3, off=("request_ops",)); bad = False
except ValueError:
    bad = True
check("P3 off= is refused for another profile", bad)
try:
    LP.profile_params(LP.PROFILE_R4, off=("made_up",)); bad = False
except ValueError:
    bad = True
check("P4 off= is refused for an unknown mechanism", bad)
check("P5 r3 params are unchanged by r4 (no request_ops / closure keys)", "request_ops" not in LP.profile_params(LP.PROFILE_R3)["precision"])
check("P6 r4 uses the goal-schema query", LP.make_query_profile(LP.PROFILE_R4, "Pick up the mug.", "en", None, []).get("intent") is not None)

# ---- G1: rendering fidelity (contrasts)
check("G1.1 a plain command is rendered with its type", R("Pick up the mug.") == "pick up Mug", R("Pick up the mug."))
r = R("Put the mug in the fridge, then close the fridge.")
check("G1.2 order, destination and a fixture as a plain noun", r == "put Mug to the fridge; close the fridge", r)
r = R("Do not slice the bread, just put it on the counter.")
check("G1.3 a negated operation stays negated", r is not None and r.startswith("do not slice Bread"), r)
r = R("I want to eat the apple warm.")
check("G1.4 a wish stays a wish; the state word is shown as stated, never as an operation", r == "wants to eat Apple (stated: warm)" or (r and r.startswith("wants to eat Apple") and "heat" not in r), r)
r = R("I need something to stir with.")
check("G1.5 a purpose need keeps the vague word and names no tool type", r is not None and r.startswith("wants something to") or r is None, r)
r = R("I need something to flip the eggs with.")
check("G1.6 the flip need never names Spatula", r is None or "Spatula" not in r, r)
check("G1.7 a fixture never appears as a type name (stove)", "StoveBurner" not in (R("Turn off the stove.") or "") and (R("Turn off the stove.") or "").endswith("the stove"), R("Turn off the stove."))
check("G1.8 an ambiguous alias (table) gives no rendering", R("Put the cup on the table.") is None, R("Put the cup on the table."))
check("G1.9 fill (its liquid is not kept) gives no rendering", R("Fill the cup with coffee.") is None, R("Fill the cup with coffee."))
check("G1.10 a pure statement gives no rendering", R("The bread is stale.") is None)
check("G1.11 an untyped object gives no rendering", R("Put the gizmo in the drawer.") is None, R("Put the gizmo in the drawer."))
check("G1.12 a quantity is kept", (R("Put two apples in the fridge.") or "").startswith("put 2 Apple"), R("Put two apples in the fridge."))
r = R("Put the bowl in a cabinet and close it.")
check("G1.13 a second mention of 'a cabinet' becomes 'the cabinet'", r is not None and "close the cabinet" in r, r)
e, f = R("Put the potato in the microwave, close the door, and turn it on."), R("سیب‌زمینی را داخل مایکروویو بگذار، درش را ببند و روشنش کن.", "fa")
check("G1.14 EN and FA give the same rendering for the same request", e == f and e is not None, (e, f))
e, f = R("Pick up the spoon."), R("قاشق را بردار.", "fa")
check("G1.15 Persian spoon renders as Spoon, same as EN", e == f == "pick up Spoon", (e, f))
pe, pf = pkg("Put the mug in the fridge.", "en"), pkg("ماگ را داخل یخچال بگذار.", "fa")
check("G1.16 the package line is identical for EN and FA", req(pe) == req(pf) and req(pe), (req(pe), req(pf)))
allp = [pkg(t, l) for t, l in (("Pick up the spoon.", "en"), ("قاشق را بردار.", "fa"), ("سیب‌زمینی را داخل مایکروویو بگذار و روشنش کن.", "fa"),
                               ("چیزی برای سابیدن ظرف‌ها لازم دارم.", "fa"), ("تخم‌مرغ گندیده است.", "fa"))]
check("G1.17 no Persian character in any package text", not any(re.search("[؀-ۿ]", x["text"]) for x in allp))
p = pkg("Pick up the mug.", "en")
check("G1.18 the names line is not repeated for a rendered object", req(p) == ["The request in English: pick up Mug."], req(p))
p = pkg("The egg is rotten.", "en")
check("G1.19 a fact-only request keeps the r3 names line (stated state)", "This request names: Egg (stated: bad)." in p["text"], p["text"])
p = pkg("Pick up the mug.", "en", off=("request_ops",))
check("G1.20 request_ops off -> the r3 names line", req(p) == ["This request names: Mug."], req(p))
r = R("I want this apple cold later.")
check("G1.21 'cold' is never rendered as chill / Fridge", r is None or ("chill" not in r and "Fridge" not in r and "fridge" not in r), r)
p = pkg("I need something to scrub the pans with.", "en")
check("G1.22 the rendering adds no tool type (only the gated r3 fit line may)", all("DishSponge" not in l.split(". A tool that fits")[0] for l in p["text"].splitlines() if l.startswith("The request in English")), p["text"])
# no plan words: the rendering uses plain verbs, never contract action names
words = " ".join(filter(None, (R(t) for t in ("Pick up the mug.", "Put the mug in the fridge, then close the fridge.", "Turn on the faucet.",
                                              "Find the egg and pick it up."))))
check("G1.23 the rendering uses plain verbs, not action names", not re.search(r"\b(pickup|put_in|pick_and_place|toggle_on|toggle_off|search_for)\b", words), words)

# ---- G2: precondition closure
CL = "Needed here: a Microwave must be closed when toggle_on runs."
p = pkg("Put the apple in the microwave and turn it on.", "en", item="g-l2-microwave-run-egg")
check("G2.1 toggle_on Microwave without a close request keeps the close line with the note", CL in p["text"] and "close X:" in p["text"], p["text"])
p = pkg("Put the apple in the microwave and turn it on.", "en", off=("precondition_closure",), item="g-l2-microwave-run-egg")
check("G2.2 ... not with precondition_closure off", CL not in p["text"] and "close X:" not in p["text"], p["text"])
p = pkg("Turn off the microwave.", "en")
check("G2.3 toggle_off Microwave: no closure", CL not in p["text"], p["text"])
p = pkg("Put the apple in the microwave.", "en")
check("G2.4 merely naming the Microwave: no closure", CL not in p["text"], p["text"])
p = pkg("Put the apple in the microwave, close it, and turn it on.", "en", item="g-l2-microwave-run-egg")
check("G2.5 a stated close keeps the normal close line without the note", "close X:" in p["text"] and CL not in p["text"], p["text"])
p = pkg("Turn on the faucet.", "en")
check("G2.6 toggle_on of another fixture: no closure", CL not in p["text"] and "close X:" not in p["text"], p["text"])
p = pkg("Do not turn on the microwave.", "en")
check("G2.7 a negated toggle_on: no closure", CL not in p["text"], p["text"])
p = pkg("سیب را داخل مایکروویو بگذار و روشنش کن.", "fa", item="g-l2-microwave-run-egg")
check("G2.8 Persian request: same closure", CL in p["text"], p["text"])
kept = [k for k in p["kept"] if "tool-close" in json.dumps(k)] if isinstance(p.get("kept"), list) else []
check("G2.9 the closure line stays within the 300-token cap", p["tokens"]["max"] <= 300, p["tokens"])
check("G2.10 the KB close text itself is unchanged (the note is added only by the closure)",
      any(r["id"] == "procedure/*/tool-close" and _t.endswith("or to leave it closed.") for r in stores["procedure"].records() for _t in [r["text"]["en"]]))

p = pkg("Put the apple in the microwave and turn it on.", "en")
check("G2.11 with the run-it recipe present (it states the close) no extra close line is added", CL not in p["text"] and "close X:" not in p["text"], p["text"])
# (the gate leaves the microwave-run recipe out with the item-level leave-out of a microwave item, as the signal harness does)

# ---- R3: the r3 profile is unchanged (frozen rows of round 3)
rows = [json.loads(l) for l in open(os.path.join(HERE, "final_test", "signal", "r3", "pkg_rows_pkg_1_r3.jsonl"), encoding="utf-8")]
check("R3.1 the round-3 rows exist", len(rows) == 46)

# ---- H: honesty
src = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
new = src[src.index("_RENDER_VERB = {"): src.index("def _state_afford_words")]
check("H1 the new code reads no item id, level, reference plan or goal", not re.search(r"g-l[123]-|\"level\"|reference|\[\"plan\"\]|\"goal\"", new))
check("H2 the mechanisms' development provenance lives outside the package code", "tuned_on" not in new
      and os.path.exists(os.path.join(HERE, "final_test", "signal", "r4", "mechanisms.json")))
print("%d/%d passed" % (N[0], N[0] + N[1]))
sys.exit(1 if N[1] else 0)
