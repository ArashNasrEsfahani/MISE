"""Gate for L-round3 (repair for round 3): F0 leave-out over the affordance records, F1 the parser-derived English request line,
F2 stated-fact affordances, F3 contract-fact tool wording. Offline (Windows python); no simulator, no LLM."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "final_test", "signal"))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ltm_package as LP, ltm_procedural as PR
from signal_pkg_plan import build_builder, LeaveOut, KITCHEN
N = [0, 0]
def check(name, ok, info=""):
    N[0 if ok else 1] += 1
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(info)) if (info and not ok) else ""))
stores, counter, _p, cats, _n = build_builder(set())
kb = stores["procedure"]
def pkg(text, lang, item=None, profile=LP.PROFILE_R3, held=None):
    q = LP.make_query_profile(profile, text, lang, held, [])
    st = dict(stores)
    if item:
        for c in ("template", "procedure", "profile", "places", "example"):
            if c in st: st[c] = LeaveOut(st[c], item)
    return LP.PackageBuilder(KITCHEN, st, counter, LP.profile_params(profile), categories=cats).build(q)
# ---- F0
full = {r["id"] for r in kb.afford_records()}
lo = {r["id"] for r in LeaveOut(kb, "g-l3-tool-flip").afford_records()}
check("F0.1 the flip rows and the flip request line exist", {"procedure/*/afford-flip", "procedure/*/afford-flip-request"} <= full)
check("F0.2 LeaveOut drops affordance rows tuned on the item (incl. the request line)", "procedure/*/afford-flip-request" not in lo and "procedure/*/afford-flip" not in lo)
check("F0.3 LeaveOut keeps rows not tuned on the item", "procedure/*/afford-discard" in lo and "procedure/*/afford-toast" in lo)
check("F0.4 an unrelated item keeps everything", {r["id"] for r in LeaveOut(kb, "g-x-none").afford_records()} == full)
p = pkg("I need something to flip the pancakes with.", "en", item="g-l3-tool-flip")
check("F0.5 the tuned flip line does not reach the item it was tuned on", "Spatula" not in p["text"], p["text"])
p = pkg("I need something to flip the pancakes with.", "en", item="g-other")
check("F0.6 ... but reaches another item", "Spatula" in p["text"], p["text"])
# ---- F1
pe, pf = pkg("Pick up the spoon.", "en"), pkg("قاشق را بردار.", "fa")
check("F1.1 the Persian spoon reaches the planner as Spoon", "This request names: Spoon." in pf["text"], pf["text"])
check("F1.2 EN and FA give the same request line", [l for l in pe["text"].splitlines() if "request" in l] == [l for l in pf["text"].splitlines() if "request" in l])
p = pkg("کفگیر را داخل یکی از کشوها بگذار و آن را ببند.", "fa")
check("F1.3 Persian spatula -> Spatula; the fixture (Drawer) is not in the request line", "This request names: Spatula." in p["text"], p["text"])
p = pkg("چنگال، قاشق و چاقو استفاده شده‌اند؛ آن‌ها را جایی بگذار که شسته می‌شوند.", "fa")
# A26 (K-B2b): the bare knife (چاقو) no longer resolves to a type, so it is not named (was "Fork, Spoon, Knife.")
check("F1.4 Persian fork/spoon/knife -> Fork, Spoon in order; the knife is not named (A26)", "This request names: Fork, Spoon." in p["text"], p["text"])
p = pkg("Turn off the stove.", "en")
check("F1.5 a fixture alias (stove) is never named in the request line", "StoveBurner" not in "".join(l for l in p["text"].splitlines() if "request" in l), p["text"])
p = pkg("چیزی برای سابیدن ظرف‌ها لازم دارم.", "fa")
check("F1.6 a purpose need shows the parser's tool type", "A tool that fits the request: DishSponge." in p["text"], p["text"])
# A26 (K-B2b): g-l3-tool-cut is archived (it no longer supports the tool template), so the leave-out is shown on g-l3-tool-flip, which
# both the cut row and the template cite; the tool is the ButterKnife (was "Knife")
p = pkg("I need something to cut with.", "en", item="g-l3-tool-flip")
check("F1.7 ... but not when that operation's KB row is left out for the item", "ButterKnife" not in p["text"], p["text"])
p = pkg("I need something to cut with.", "en", item="g-other")
check("F1.8 ... and it does for another item", "ButterKnife" in p["text"], p["text"])
p1 = pkg("Pick up the spoon.", "en", profile=LP.PROFILE_R1)
check("F1.9 profile r1 has no request line (round-2 behaviour kept)", "This request names" not in p1["text"])
allp = [pkg(t, l) for t, l in (("Pick up the spoon.", "en"), ("قاشق را بردار.", "fa"), ("تخم‌مرغ گندیده است.", "fa"), ("چیزی برای سابیدن ظرف‌ها لازم دارم.", "fa"))]
check("F1.10 no Persian character in any package text", not any(re.search("[؀-ۿ]", x["text"]) for x in allp))
p = pkg("I do not want the spoon.", "en")
check("F1.11 the request line is a type list only (no verb, no step)", all(not re.search(r"\b(pickup|put_in|pick_and_place|toggle)", l) for l in p["text"].splitlines() if l.startswith("This request")))
# ---- F2
pe, pf = pkg("The egg is rotten.", "en", item="g-l3-rotten-egg"), pkg("تخم‌مرغ گندیده است.", "fa", item="g-l3-rotten-egg")
check("F2.1 a stated bad state reaches the existing discard line (EN)", "To throw something away put it in the GarbageCan." in pe["text"], pe["text"])
check("F2.2 ... (FA, same text)", pe["text"] == pf["text"], (pe["text"], pf["text"]))
check("F2.3 the state is shown as the parser read it", "Egg (stated: bad)" in pe["text"])
p = pkg("The egg is rotten.", "en", profile=LP.PROFILE_R1, item="g-l3-rotten-egg")
check("F2.4 profile r1 keeps the round-2 (empty) package", p["empty"], p["text"])
p = pkg("I want to eat this apple cold later.", "en", item="g-l3-chilled-apple")
check("F2.5 a recipe AND its row both tuned on the item stay out (no Fridge advice)", "Fridge" not in p["text"], p["text"])
p3, p1 = pkg("It is too dark in here.", "en"), pkg("It is too dark in here.", "en", profile=LP.PROFILE_R1)
check("F2.6 a state with no KB row (dark -> LightSwitch) adds nothing beyond r1 (no affordance line made up)", [l for l in p3["text"].splitlines() if not l.startswith("This request")] == p1["text"].splitlines(), (p3["text"], p1["text"]))
check("F2.7 _state_afford_words: no word for a negated or op-bearing key", LP._state_afford_words(kb, {"state": "bad", "op": "put"}) == () and LP._state_afford_words(kb, {"state": "bad", "neg": True}) == ())
# ---- F3 (contract facts only)
tt = {e["id"]: e["text"]["en"] for e in kb.entries} if hasattr(kb, "entries") else {k: v["text"]["en"] for k, v in kb.by_id.items()}
check("F3.1 toggle_on: the ambiguous 'close Microwave first' order clause is gone", "close Microwave first" not in tt["proc:tool-toggle-on"])
check("F3.2 toggle_on still states the closed-door fact", "only switches on while its door is closed" in tt["proc:tool-toggle-on"])
check("F3.3 put_in and pick_and_place say the door may stay open", "may leave its door open" in tt["proc:tool-put-in"] and "may leave its door open" in tt["proc:tool-pick-and-place"])
check("F3.4 open: an explicit open stays open until a close step", "stays open until a close X step" in tt["proc:tool-open"])
check("F3.5 no order advice was written (it would be tuned on the microwave items)", not any("then close Microwave, then toggle_on" in v for v in tt.values() if v))
src = open(os.path.join(HERE, "honest_skills.py"), encoding="utf-8").read()
check("F3.6 contract basis: tidy never closes an explicit open; the step-end close skips needed-later doors", 'e["why"] != "explicit"' in src and "_needed_later(e)" in src)
check("F3.7 KB contract truth check still clean", not PR.check_skill_truth(kb) if hasattr(PR, "check_skill_truth") else True)
# ---- honesty of the new code
new = open(os.path.join(HERE, "ltm_package.py"), encoding="utf-8").read()
seg = new[new.index("def _request_rec"): new.index("def _procedure_with_afford")] + new[new.index("def _state_afford_words"): new.index("class PackageBuilder")]
seg = re.sub(r'""".*?"""', "", seg, flags=re.S)                     # docstrings may say what the code does NOT read
check("H1 the new code reads no item id, level, reference plan or goal", not re.search(r"\bg-l[123]|\blevel\b|reference_plan|\[\"plan\"\]|\bgoal_ok\b", seg))
print("%d/%d passed" % (N[0], sum(N)))
sys.exit(1 if N[1] else 0)
