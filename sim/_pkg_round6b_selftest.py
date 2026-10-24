"""Gate for K-round6b (stage K-r6b build): profile pkg-1.r6b = pkg-1.r6 + need_hand (a placeholder NEED whose placeholder is a THING
word gets NEED_THING_INSTRUCTION INSTEAD OF NEED_INSTRUCTION; everything else is pkg-1.r6). The ten gate groups of
docs/final-test/kitchen_memory/ROUND6B-FIXES.md section 4. Offline (Windows python, repo root); no simulator, no LLM call. Gate sentences
are general kitchen language written for the gate; the set blocks rebuild the frozen round-6 dev rows (original 20, fresh 22, need checks)."""
import hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SIG = os.path.join(HERE, "final_test", "signal"); R6 = os.path.join(SIG, "r6"); R6B = os.path.join(SIG, "r6b")
sys.path.insert(0, HERE); sys.path.insert(0, SIG); sys.path.insert(0, R6); sys.path.insert(0, R6B)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ltm_package as LP, goal_schema as GS, type_scope as TS
from signal_pkg_plan import build_builder, LeaveOut, KITCHEN
from signal_plan import holding_of
N = [0, 0]


def check(name, ok, info=""):
    N[0 if ok else 1] += 1
    print(("PASS " if ok else "FAIL ") + name + (("  " + str(info)[:600]) if (info and not ok) else ""))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


stores, counter, _p, cats, _n = build_builder(set())


def pkg(text, lang="en", profile=LP.PROFILE_R6B, held=None, off=(), item=None, summary=(), prm=None):
    q = LP.make_query_profile(profile, text, lang, held, list(summary))
    st = dict(stores)
    if item:
        for c in ("template", "procedure", "profile", "places", "example"):
            if c in st: st[c] = LeaveOut(st[c], item)
    prm = prm or LP.profile_params(profile, off=off)
    return LP.PackageBuilder(KITCHEN, st, counter, prm, categories=cats).build(q)


MARK = "Kitchen inventory: the homeowner reports"


def bundle(p):
    t = p["text"]; i = t.find(MARK)
    return t[i:] if i >= 0 else ""


THING, R6LINE = LP.NEED_THING_INSTRUCTION, LP.NEED_INSTRUCTION
USES = json.load(open(LP.USES_PATH, encoding="utf-8"))
VOCAB = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]
AMEND = json.load(open(os.path.join(R6B, "amendment_r6b.json"), encoding="utf-8"))
MECH = json.load(open(os.path.join(R6B, "mechanisms.json"), encoding="utf-8"))
MECH6 = json.load(open(os.path.join(R6, "mechanisms.json"), encoding="utf-8"))

# ---- 0: the profile
p6b = LP.profile_params(LP.PROFILE_R6B)
pr = p6b["precision"]
check("0.1 r6b is a known profile; R6B_MECHANISMS = R6 + need_hand; every mechanism on, D2 filter on",
      LP.PROFILE_R6B in LP.PROFILES and LP.R6B_MECHANISMS == LP.R6_MECHANISMS + ("need_hand",)
      and all(pr.get(m) for m in LP.R6B_MECHANISMS) and pr.get("type_scope"))
try:
    LP.profile_params(LP.PROFILE_R6, off=("need_hand",)); bad = False
except ValueError:
    bad = True
check("0.2 need_hand is valid only for pkg-1.r6b (r6 refuses it)", bad and "need_hand" not in LP.profile_params(LP.PROFILE_R6)["precision"])
check("0.3 the exact accepted wording, ASCII, frozen", THING == ("Select a suitable in-scope item and get it into your hand. pickup finds and takes the item and "
      "requires an empty hand; search_for does not pick it up. Do not perform the item's intended use unless the request explicitly asks you to.")
      and THING.isascii() and "’" not in THING)
check("0.4 the pre-Astra draft is gone (no NEED_HOLD_LINE, no need_keep_hints / need_hold_line anywhere in ltm_package)",
      not hasattr(LP, "NEED_HOLD_LINE") and "need_keep_hints" not in open(LP.__file__, encoding="utf-8").read()
      and "need_hold_line" not in open(LP.__file__, encoding="utf-8").read())

# ---- 1: routing (section 2 table) on synthetic EN / FA cases
cases = [  # (text, lang, expected route: thing | place | desired | none)
    ("I need something to stir the soup with.", "en", "thing"), ("I'd like something to eat.", "en", "thing"),
    ("I want something to drink.", "en", "thing"), ("چیزی برای هم زدن سوپ لازم دارم.", "fa", "thing"),
    ("یه چیزی می‌خوام که باهاش میز را پاک کنم.", "fa", "thing"), ("یک چیز برای هم زدن سوپ لازم دارم.", "fa", "thing"),
    ("I need somewhere to put my keys.", "en", "place"), ("I need someplace to put my keys.", "en", "place"),
    ("I need anywhere to sit.", "en", "place"), ("جایی لازم دارم که کلیدهایم را بگذارم.", "fa", "place"),
    ("جایی برای گذاشتن لیوان لازم دارم.", "fa", "place"),
    ("I want this tomato warm.", "en", "desired"), ("I want to eat this bread hot later.", "en", "desired"),
    ("I'd like the mug clean.", "en", "desired"), ("گوجه را گرم می‌خواهم.", "fa", "desired"),
    ("Pick up the spoon.", "en", "none"), ("Put the apple in the fridge and close the door.", "en", "none"), ("قاشق را بردار.", "fa", "none"),
    ("کاسه را داخل یکی از کابینت‌ها بگذار و درش را ببند.", "fa", "none"),
    ("I don't need anything to stir with.", "en", "none"), ("I need nothing to stir with.", "en", "none"),
    ("چیزی برای هم زدن لازم ندارم.", "fa", "none"),
    ("Do I need something to stir with?", "en", "none"), ("آیا چیزی برای هم زدن لازم دارم؟", "fa", "none"),
    ("I like warm tomatoes.", "en", "none"), ("Bring me something to stir with.", "en", "none")]
for text, lang, route in cases:
    b6b, b6 = pkg(text, lang), pkg(text, lang, profile=LP.PROFILE_R6)
    tr = b6b["need_rule"]["trigger"]
    got = ("thing" if tr.get("placeholder") == "thing" else "place") if tr.get("kind") == "placeholder" else \
          ("desired" if tr.get("kind") == "desired_state" else "none")
    if route == "thing":
        ok = got == "thing" and THING in b6b["text"] and b6b["need_rule"]["instruction"] == "thing"
    else:
        ok = got == route and THING not in b6b["text"] and b6b["text"] == b6["text"] and \
             ((R6LINE in b6b["text"]) == (route in ("place", "desired")))
    check("1 route %-7s %s: %s" % (route, lang, text), ok, (got, tr, b6b["text"][-300:]))
check("1.b the r6 verdict is unchanged by r6b (need_trigger_r6b = need_trigger + placeholder fields only)",
      all({k: v for k, v in GS.need_trigger_r6b(t, l).items() if k not in ("placeholder", "placeholder_lex")} == GS.need_trigger(t, l)
          for t, l, _r in cases))
check("1.c need_trigger (r6) returns no placeholder field (r6 output byte-identical)",
      all("placeholder" not in GS.need_trigger(t, l) and "placeholder_lex" not in GS.need_trigger(t, l) for t, l, _r in cases))
st_off = pkg("I need something to stir the soup with.", off=("need_hand",))
st_r6 = pkg("I need something to stir the soup with.", profile=LP.PROFILE_R6)
check("1.d need_hand off: a placeholder-THING NEED gets the r6 instruction line (text = pkg-1.r6)",
      st_off["text"] == st_r6["text"] and THING not in st_off["text"] and st_off["need_rule"]["instruction"] == "r6")

# ---- 2: replacement, not append
for text, lang in (("I need something to stir the soup with.", "en"), ("چیزی برای هم زدن سوپ لازم دارم.", "fa")):
    b = bundle(pkg(text, lang))
    check("2 %s: THING line exactly once, the r6 line zero times, last line of the bundle" % lang,
          b.count(THING) == 1 and b.count(R6LINE) == 0 and b.splitlines()[-1] == THING
          and pkg(text, lang)["text"].count(THING) == 1)

# ---- 3: same bytes everywhere (EN / FA, gemma / anthropic builders)
ea, fa_ = pkg("I need something to stir the soup with."), pkg("چیزی برای هم زدن سوپ لازم دارم.", "fa")
check("3.1 the bundle (incl. the line) is identical for EN and FA", bundle(ea) == bundle(fa_) and bundle(ea).endswith(THING))
check("3.2 the line is identical for different thing needs", bundle(pkg("I need something to read while I wait.")) == bundle(ea))
try:
    import planners as P
    import honest_skills as hs
    from planners.anthropic_provider import AnthropicPlanner
    types = TS.filter_types(VOCAB)
    con = P.PlannerContract(types, tools=hs.TOOLS, version="ft-v2")
    msg_g = con.messages("I need something to stir the soup with.", [], None, memory_block=ea["text"])
    body_o = AnthropicPlanner(P.ARMS["opus"][1]).build_body(con.messages("I need something to stir the soup with.", [], None, memory_block=ea["text"]),
                                                            {"type": "object", "properties": {}})
    ug = [m["content"] for m in msg_g if m["role"] == "user"][0]
    uo = [m["content"] for m in body_o["messages"] if m["role"] == "user"][0]
    check("3.3 Gemma+M and Opus+M receive the same user text with the line once (provider builders, no call)",
          ug == uo and ug.count(THING) == 1 and ug.count(R6LINE) == 0)
except Exception as e:                                                      # pragma: no cover
    check("3.3 Gemma+M and Opus+M receive the same user text with the line once (provider builders, no call)", False, repr(e))
LEX6B_RAW = json.load(open(LP.NEED_CUES_R6B_PATH, encoding="utf-8"))
check("3.4 language differences are confined to the trigger lexicon: one constant, no per-language variant in ltm_package",
      open(LP.__file__, encoding="utf-8").read().count("NEED_THING_INSTRUCTION = ") == 1
      and not any(re.search(r"NEED_THING_INSTRUCTION_(EN|FA)", open(LP.__file__, encoding="utf-8").read()) for _ in [0]))

# ---- 4: nothing hidden in the line
types_all = set(VOCAB) | {r["type"] for r in USES["rows"]} | set(TS.EXCLUDED_TYPES)
check("4.1 no type name in the line (detector types + uses-table types + excluded, case-insensitive whole word)",
      not [t for t in types_all if re.search(r"(?i)(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(t), THING)])
check("4.2 no item id in the line", not re.search(r"g-l[123]-", THING))
PRED = ("isPickedUp", "parentReceptacles", "isOpen", "isToggled", "isSliced", "isCooked", "isDirty", "isFilledWithLiquid", "isBroken",
        "isUsedUp", "temperature", "receptacleObjectIds", "isPickedup")
check("4.3 no goal predicate key in the line", not [k for k in PRED if k.lower() in THING.lower()])
uses_words = {w for r in USES["rows"] for w in re.findall(r"[a-z]+", r["use"]) if len(w) > 4}
uses_phr = [ph.strip() for r in USES["rows"] for ph in r["use"].split(";")]
overlap = set(re.findall(r"[a-z]+", THING.lower())) & uses_words
check("4.4 no purpose-to-type map: no '->' / ':' mapping, no uses-table purpose phrase, and the only shared long word is the "
      "contract verb 'takes' (pickup takes the item; the TrashCan row 'takes rubbish' is unrelated)",
      "->" not in THING and ":" not in THING and not [ph for ph in uses_phr if ph.lower() in THING.lower()] and overlap <= {"takes"}, overlap)
check("4.5 the only tool names in the line are pickup and search_for (contract tools)",
      set(re.findall(r"\b[a-z]+_[a-z_]+\b|\bpickup\b", THING)) == {"pickup", "search_for"})

# ---- 5: the bundle is otherwise unchanged
b = bundle(ea)
lines = b.splitlines()
check("5.1 all 48 rows in alphabetical order, Knife absent", [l.split(":")[0] for l in lines[1:-2]] == [r["type"] for r in USES["rows"]]
      and len(USES["rows"]) == 48 and not re.search(r"(?m)^Knife:", b) and len(lines) == 48 + 3)
check("5.2 'Seen by you so far' present; the bundle equals the r6 bundle up to the last line",
      "Seen by you so far: none yet." in b and lines[:-1] == bundle(st_r6).splitlines()[:-1])
allp = [LP.profile_params(LP.PROFILE_R6B)] + [LP.profile_params(LP.PROFILE_R6B, off=(m,)) for m in LP.R6B_MECHANISMS]
check("5.3 need_keep_hints / need_hold_line absent (or False) in every r6b params variant",
      all(not x["precision"].get("need_keep_hints") and not x["precision"].get("need_hold_line") for x in allp))
q6b = LP.make_query_profile(LP.PROFILE_R6B, "I want to eat this bread hot later.", "en", None, [])
check("5.4 strip_need_hints active: a NEED package has no 'A tool that fits' line and an empty route audit",
      "A tool that fits" not in ea["text"] and ea["need_audit"] == [] and pkg("I want to eat this bread hot later.")["need_audit"] == [])

# ---- sets: rebuild r6 (frozen) and r6b live
from pkg_r6 import sets, make_build as mk6
import pkg_r6b as PB
man, by, pres = sets()
st0, cnt0, _x, cats0, _y = build_builder({p["item"] for p in pres})
b6, b6b = mk6(st0, cnt0, cats0, by), PB.make_build(st0, cnt0, cats0, by)
rows6, rows6b = {}, {}
for cond in ("universal", "strict"):
    for p in pres:
        rows6[(cond, p["item"], p["lang"])] = b6(p, cond)
        rows6b[(cond, p["item"], p["lang"])] = b6b(p, cond)
scrub = rows6b[("universal", "g-l3-tool-scrub", "fa")]
check("5.5 tool-scrub FA r6b package: no 'A tool that fits', NEED active, THING line, empty route audit",
      "A tool that fits" not in scrub["text"] and scrub["need"]["active"] and THING in scrub["text"] and scrub["need_audit"] == [])
same_drop = all(sorted(d for d in r6b_["kept"]) == sorted(rows6[k]["kept"]) for k, r6b_ in rows6b.items())
check("5.6 the route audit and the hint strip act as in r6: the same records kept on every presentation (both conditions)", same_drop)

# ---- 6: byte identity
nonneed = [k for k, r in rows6b.items() if not r["need"]["active"]]
other_need = [k for k, r in rows6b.items() if r["need"]["active"] and r["need"]["instruction"] != "thing"]
thing = [k for k, r in rows6b.items() if r["need"]["instruction"] == "thing"]
check("6.1 every non-NEED r6b row equals pkg-1.r6 (%d rows)" % len(nonneed), nonneed and all(rows6b[k]["text"] == rows6[k]["text"] for k in nonneed))
check("6.2 every place / desired-state NEED row equals pkg-1.r6 (%d rows)" % len(other_need),
      all(rows6b[k]["text"] == rows6[k]["text"] for k in other_need) and any(rows6b[k]["need"]["kind"] == "desired_state" for k in other_need))
check("6.3 every placeholder-THING row = the r6 text with the line replaced once (%d rows)" % len(thing),
      thing and all(rows6[k]["text"].count(R6LINE) == 1 and rows6b[k]["text"] == rows6[k]["text"].replace(R6LINE, THING) for k in thing))
# non-NEED rows equal r4 after the same D2 transformation (type_scope on), not raw pre-D2 r4
eq4, n4 = 0, 0
for k in nonneed:
    cond, item, lang = k
    it = by[item]
    off4 = tuple(m for m in rows6b[k]["off"] if m in LP.R4_MECHANISMS)
    p4 = LP.profile_params(LP.PROFILE_R4, off=off4)
    p4["precision"]["type_scope"] = True
    p4["sha256"] = LP.params_sha(p4)
    st = dict(st0)
    for c in ("template", "procedure", "profile", "places", "example"):
        if c in st: st[c] = LeaveOut(st[c], item)
    q4 = LP.make_query_profile(LP.PROFILE_R4, it[lang], lang, holding_of(it), [])
    n4 += 1
    eq4 += LP.PackageBuilder(KITCHEN, st, cnt0, p4, categories=cats0).build(q4)["text"] == rows6b[k]["text"]
check("6.4 non-NEED r6b rows equal r4 + the D2 filter (%d of %d)" % (eq4, n4), eq4 == n4 and n4 > 0)
for cond, fsha in (("universal", "a5e8319b7027dc54ca95f31f06fb978f22133e27cbdcba98c21ded1e74a13bcb"),
                   ("strict", "8ff3c134b0da6e0f563e39b81c4b68d912756cfa8ef2d2369390020b3521d9da")):
    blob = "".join(json.dumps(rows6[(cond, p["item"], p["lang"])], ensure_ascii=False) + "\n" for p in pres).encode("utf-8")
    fn = os.path.join(R6, "pkg_rows_pkg_1_r6_%s.jsonl" % cond)
    # A26 (K-B2b 2026-10-13) replaces: "6.5 frozen r6 rows rebuild unchanged" (blob sha == frozen sha). On the A26 successor version the
    # frozen r6 file is unchanged on disk, and a rebuild differs from it ONLY in the egg FA presentation (decision 8 fixed its text).
    _old6 = {(r["item"], r["lang"]): r for r in map(json.loads, open(fn, encoding="utf-8"))}
    _chg6 = sorted(k for k in _old6 if _old6[k]["text"] != rows6[(cond,) + k]["text"])
    check("6.5 frozen r6 %s rows: file unchanged (sha %s); a successor rebuild differs only in the egg FA text (%s)" % (cond, fsha[:8], _chg6),
          sha(fn) == fsha and _chg6 == [("g-l2-microwave-run-egg", "fa")])
_SET = json.load(open(os.path.join(R6B, "exec_settings_r6b.json"), encoding="utf-8"))
for cond in ("universal", "strict"):
    # A26 (K-B2b): the rows the r6b check uses are the RE-FROZEN successor rows named in exec_settings_r6b.json (pkg_r6b_kb2b.py);
    # the pre-K-B2b r6b rows stay on disk as history and differ from the successor only in the egg FA row
    fn = os.path.join(ROOT, (_SET.get("package_rows") or {}).get(cond) or "sim/final_test/signal/r6b/pkg_rows_pkg_1_r6b_kb2b_%s.jsonl" % cond)
    blob = "".join(json.dumps(rows6b[(cond, p["item"], p["lang"])], ensure_ascii=False) + "\n" for p in pres).encode("utf-8")
    check("6.6 the re-frozen r6b %s rows on disk equal a fresh rebuild" % cond, os.path.exists(fn) and blob == open(fn, "rb").read())
    _pre = {(r["item"], r["lang"]): r for r in map(json.loads, open(os.path.join(R6B, "pkg_rows_pkg_1_r6b_%s.jsonl" % cond), encoding="utf-8"))}
    _chg = sorted(k for k in _pre if _pre[k]["text"] != rows6b[(cond,) + k]["text"])
    check("6.6b the pre-K-B2b r6b %s rows differ from the successor only in the egg FA row (%s)" % (cond, _chg), _chg == [("g-l2-microwave-run-egg", "fa")])

# ---- 7: STRICT
tuned = ["g-l3-tool-flip", "g-l3-tool-scrub", "g-l3-need-read", "g-l3-need-fry-egg"]
check("7.1 need_hand tuned_on = the 4 items of section 3, for EN and FA", MECH["need_hand"]["tuned_on"] == tuned and MECH["need_hand"]["languages"] == ["en", "fa"])
ok71 = all(("need_hand" in rows6b[("strict", i, l)]["off"]) for i in tuned for l in ("en", "fa"))
ok72 = all(("need_rule" in rows6b[("strict", i, l)]["off"]) for i in ("g-l3-tool-flip", "g-l3-chilled-apple") for l in ("en", "fa"))
check("7.2 STRICT union mask: need_hand off on the 4 tuned items, need_rule off on tool-flip / chilled-apple, both languages", ok71 and ok72)
check("7.3 earlier masks unchanged: r6b strict off minus need_hand = r6 strict off on every presentation",
      all([m for m in rows6b[("strict", p["item"], p["lang"])]["off"] if m != "need_hand"] == rows6[("strict", p["item"], p["lang"])]["off"] for p in pres)
      and all({k: v for k, v in MECH.items() if k in MECH6} == MECH6 for _ in [0]))
check("7.4 STRICT: no THING line on any tuned item; tool-flip / chilled-apple keep their r4 fallback (no bundle)",
      all(THING not in rows6b[("strict", i, l)]["text"] for i in tuned for l in ("en", "fa"))
      and all(not rows6b[("strict", i, l)]["need"]["active"] for i in ("g-l3-tool-flip", "g-l3-chilled-apple") for l in ("en", "fa")))
PM = json.load(open(os.path.join(R6B, "provenance_matrix.json"), encoding="utf-8"))
pm_ok = len(PM["rows"]) == len(pres) and all(set(r) >= set(LP.R6B_MECHANISMS) | {"tidy_switch_closed"} for r in PM["rows"]) and all(
    r["need_hand"]["active_strict"] == (r["item"] not in tuned) for r in PM["rows"])
check("7.5 provenance matrix r6b/provenance_matrix.json: every presentation x mechanism, need_hand strict mask as declared", pm_ok)
check("7.6 the matrix is frozen before planning: no r6b plans file exists; its sha is in the amendment freeze",
      not os.path.exists(os.path.join(R6B, "plans_r6b_gp.jsonl"))
      and (AMEND.get("freeze") or {}).get("sim/final_test/signal/r6b/provenance_matrix.json") == sha(os.path.join(R6B, "provenance_matrix.json")))

# ---- 8: contract truth
EC = json.load(open(os.path.join(HERE, "planners", "exec_contract.json"), encoding="utf-8"))["tools"]
check("8.1 pickup ends_in_hand and requires_empty_hand (and walks / opens itself)", EC["pickup"]["ends_in_hand"] is True and EC["pickup"]["requires_empty_hand"] is True
      and EC["pickup"]["walks_itself"] is True and EC["pickup"]["opens_itself"] is True)
check("8.2 search_for does not end in hand", EC["search_for"]["ends_in_hand"] is False)
check("8.3 wash does not end in hand (the item is left in the sink)", EC["wash"]["ends_in_hand"] is False)
# A26 (K-B2b) replaces: the R6-D1D2 pin 48f7505e... (pre-K-B2b). The contract is the regenerated A26 successor, pinned by the r6b re-freeze.
# BK-LINE 2026-10-13 (Astra BK-LINE-closed): once round 6b is CLOSED (run, result recorded), an approved later change of a pinned
# file is allowed only as a REGISTERED drift (sim/final_test/signal/r6b/post_check_drift.json: frozen sha, archived frozen bytes,
# approved live sha, stage). The pre-run assertions are kept verbatim for the not-CLOSED state.
sys.path.insert(0, R6B)
import r6b_closed as RC
CLOSED = RC.closed_record() is not None
REG = (RC.load_register().get("entries") or {}) if CLOSED else {}
_ECP = "sim/planners/exec_contract.json"
_ec_live = sha(os.path.join(HERE, "planners", "exec_contract.json"))
_ec_frozen = (AMEND.get("freeze") or {}).get(_ECP)
check("8.4 the contract file is the one pinned by the r6b re-freeze (fails if a contract changes under the line); not the pre-K-B2b one"
      + (" -- CLOSED: or the registered approved post-run bytes, with the frozen bytes archived" if CLOSED else ""),
      _ec_frozen != "48f7505eacfb5c34446cc0cec4282d75d3c2e6f0a16f3cf8a2cbf5e14c74910a"
      and (_ec_live == _ec_frozen or (CLOSED and _ECP in REG and REG[_ECP]["approved_sha"] == _ec_live
                                      and REG[_ECP]["frozen_sha"] == _ec_frozen
                                      and sha(os.path.join(ROOT, REG[_ECP]["archive"])) == _ec_frozen)))
c1 = (AMEND.get("protocol_choices") or {}).get("R6b-C1") or {}
check("8.5 'in hand' is recorded as protocol choice R6b-C1, a declared task convention, not a contract",
      "declared task convention" in c1.get("kind", "") and "NOT a contract" in c1.get("kind", ""))
check("8.6 the authorization is recorded and still PENDING (no confirmation invented)",
      AMEND["authorization"]["status"] in ("PENDING", "CONFIRMED") and (AMEND["authorization"]["status"] != "CONFIRMED" or AMEND["authorization"]["confirmed_date"]))

# ---- 9: lexicon
check("9.1 the r6 lexicon is untouched (sha 80a22c35...)", sha(GS.NEED_LEX_PATH) == "80a22c35d1427fa3f4c0e1724e09bed64711500d6baad508b1bb29fc59f92fbe")
L6 = json.load(open(GS.NEED_LEX_PATH, encoding="utf-8"))
check("9.2 the r6b lexicon is a copy of r6-v1 plus the two lists, version r6b-v1",
      LEX6B_RAW["version"] == "r6b-v1" and all(LEX6B_RAW[l][k] == v for l in ("en", "fa") for k, v in L6[l].items())
      and all(set(LEX6B_RAW[l]) - set(L6[l]) == {"placeholder_thing", "placeholder_place"} for l in ("en", "fa"))
      and all(LEX6B_RAW[k] == L6[k] for k in ("schema", "meaning", "written")))
check("9.3 thing and place lists are disjoint and cover every r6 placeholder token, per language",
      all(not (set(LEX6B_RAW[l]["placeholder_thing"]) & set(LEX6B_RAW[l]["placeholder_place"]))
          and set(LEX6B_RAW[l]["placeholder_thing"]) | set(LEX6B_RAW[l]["placeholder_place"]) == set(L6[l]["placeholder"]) for l in ("en", "fa")))
check("9.4 the declared token lists (section 2)", LEX6B_RAW["en"]["placeholder_thing"] == ["something", "anything"]
      and LEX6B_RAW["en"]["placeholder_place"] == ["somewhere", "someplace", "anywhere"]
      and LEX6B_RAW["fa"]["placeholder_thing"] == ["چیزی", "چیز"] and LEX6B_RAW["fa"]["placeholder_place"] == ["جایی"])
lx = GS.need_lexicon_r6b()
check("9.5 every token classifies by itself (EN and FA symmetric)", all(w in lx["en"]["thing"] for w in ("something", "anything"))
      and all(w in lx["en"]["place"] for w in ("somewhere", "someplace", "anywhere")) and len(lx["fa"]["thing"]) == 2 and len(lx["fa"]["place"]) == 1)
lexraw6b = open(LP.NEED_CUES_R6B_PATH, encoding="utf-8").read()
check("9.6 the r6b lexicon has no type name and no item id", not re.search(r"g-l[123]-", lexraw6b)
      and not any(re.search(r"\b%s\b" % re.escape(t), lexraw6b) for t in VOCAB))
check("9.7 the r6b lexicon is frozen with its own sha (params + amendment freeze)",
      pr["need_cues_r6b_sha"] == sha(LP.NEED_CUES_R6B_PATH) == (AMEND.get("freeze") or {}).get("sim/step5/lexicon/need_cues_r6b.json"))

# ---- 10: no prohibited data
Sf = json.load(open(os.path.join(ROOT, "docs/final-test/kitchen_memory/signal_set_r2_fresh.json"), encoding="utf-8"))
forbid = set(man["test"]) | set(man.get("test_uncapped_not_scored") or []) | set(Sf["acceptance_reservation"]["items"])
texts = {fn: open(os.path.join(R6B, fn), encoding="utf-8").read() for fn in os.listdir(R6B) if fn.endswith((".json", ".jsonl", ".py"))}
hit = sorted({(fn, i) for fn, t in texts.items() for i in forbid if re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(i), t)})
check("10.1 no test / test_uncapped / reserved-acceptance id in any r6b manifest, row file or script (%d forbidden ids)" % len(forbid), not hit and len(forbid) > 0, hit)
check("10.2 no test / acceptance item among the r6b presentations", not ({p["item"] for p in pres} & forbid))
check("10.3 FloorPlan15 / FloorPlan17 absent from every r6b file; kitchen is FloorPlan1",
      not any(re.search(r"FloorPlan1[57]", t) for t in texts.values()) and KITCHEN == "FloorPlan1")
check("10.4 no test / acceptance id in the r6b lexicon", not any(i in lexraw6b for i in forbid))

# ---- freeze (when recorded): every frozen artifact still matches
fz = AMEND.get("freeze") or {}
bad = [k for k, v in fz.items() if not os.path.exists(os.path.join(ROOT, k)) or sha(os.path.join(ROOT, k)) != v]
if not CLOSED:
    check("F the amendment freeze map holds (%d artifacts)" % len(fz), fz and not bad, bad)
else:   # replaces (pre-run): check("F the amendment freeze map holds", fz and not bad, bad)
    _cc = RC.check_closed()
    check("F CLOSED: every pin of the freeze map (%d artifacts) holds, or is a registered approved drift with archived frozen bytes "
          "(%d registered; the r6b gate refuses on exactly those, so round 6b is not re-run)" % (len(fz), len(REG)),
          fz and set(bad) == set(REG) and not _cc, {"unregistered": sorted(set(bad) - set(REG)), "problems": _cc[:3]})

print("pkg_round6b selftest: %d/%d" % (N[0], N[0] + N[1]))
sys.exit(0 if not N[1] else 1)
