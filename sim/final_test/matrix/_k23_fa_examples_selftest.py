"""Offline gate for k23_fa_examples.py (Persian-practised practice examples rendered in English under Arash's decision record)."""
import contextlib, io, json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import k23_fa_examples as FAX
import ltm_examples as LE
import capture as CP
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def fa(s):
    return any("؀" <= ch <= "ۿ" for ch in s or "")


ROOT = C.paths()["root"]
items = {i["id"]: i for i in json.load(open(C.paths()["dataset"], encoding="utf-8"))["items"]}
IT = "g-l3-serve-hot-potato"
plan = [{"action": "pickup", "object": "Potato"}, {"action": "heat", "object": "Potato"}, {"action": "pick_and_place", "object": "Potato", "to": "Plate"}]
steps = [{"ok": True}] * 3


def row(lang, attempt, item=IT):
    return {"namespace": "practice-k3-v1", "order": 0, "attempt": attempt, "item": item, "lang": lang, "instruction": items[item][lang],
            "plan": [[s["action"], s["object"], s.get("to"), None] for s in plan], "goal_ok": True, "status": "executed",
            "exec_steps": [{"action": s["action"], "object": s["object"], "to": s.get("to"), "ok": True} for s in plan], "key": f"k{attempt}",
            "admission": True}


rows = [row("fa", 3), row("en", 4, "g-l1-place-mug-counter-top")]
corpus = LE.build_corpus(rows, "practice-kitchen", guard=False)
idx = LE.ExampleIndex(corpus)
ex_fa = next(e for e in corpus if e["lang"] == "fa")
ex_en = next(e for e in corpus if e["lang"] == "en")
orig_render = LE.ExampleIndex.render
orig_exempt = CP.exempt_spans

# ---- 1 the defect without a decision (nothing patched)
fx = tempfile.mkdtemp(prefix="k23fax_")
os.makedirs(os.path.join(fx, "sim", "dataset")); shutil.copy(C.paths()["dataset"], C.paths(fx)["dataset"])
check("no decision record -> problems, install() patches nothing", FAX.decision_problems(fx) and FAX.install(fx) is False
      and LE.ExampleIndex.render is orig_render and CP.exempt_spans is orig_exempt)
en_line = idx.to_rec(ex_fa)["text"]["en"]
check("the defect reproduced: a Persian-practised example's ENGLISH package line quotes Persian", fa(en_line) and "Example from my practice" in en_line)

# ---- 2 decision record validation
os.makedirs(os.path.dirname(FAX.decision_path(fx)))
good = {"schema": FAX.SCHEMA, "id": FAX.DECISION_ID, "choice": FAX.CHOICE, "rule": FAX.RULE, "by": "Arash", "date": "2026-10-20", "said": "x"}
good["sha256"] = C.self_sha(good)
for name, d in (("wrong choice", dict(good, choice="drop")), ("not Arash", dict(good, by="someone")), ("no words", dict(good, said=" "))):
    d = dict(d); d["sha256"] = C.self_sha(d)
    json.dump(d, open(FAX.decision_path(fx), "w"))
    check(f"decision refused: {name}", FAX.decision_problems(fx) != [])
json.dump(dict(good, said="edited"), open(FAX.decision_path(fx), "w"))
check("decision refused: edited after writing (self-sha)", any("self-sha" in p for p in FAX.decision_problems(fx)))
json.dump(good, open(FAX.decision_path(fx), "w"))
check("a valid decision record", FAX.decision_problems(fx) == [])

# ---- 3 install + rendering
check("install() activates with the decision", FAX.install(fx) is True and getattr(LE.ExampleIndex.render, "k23_fa_examples", False)
      and getattr(CP.exempt_spans, "k23_fa_examples", False))
check("install() is idempotent", FAX.install(fx) is True)
rec = idx.to_rec(ex_fa)
check("the English line of a Persian-practised example quotes the item's registered English instruction",
      not fa(rec["text"]["en"]) and ('"%s"' % items[IT]["en"]) in rec["text"]["en"] and rec["text"]["en"].startswith("Example from my practice (succeeded)"))
check("its plan / outcome part is unchanged", rec["text"]["en"].endswith(en_line.split(" -> ", 1)[1]))
check("the Persian rendering (text fa, never used by the English package) is unchanged", items[IT]["fa"] in rec["text"]["fa"])
check("retrieval inputs unchanged: the record's lex still carries the Persian instruction on the fa side", items[IT]["fa"] in rec["lex"]["fa"])
check("an English-practised example is unchanged", idx.to_rec(ex_en)["text"]["en"] == orig_render(idx, ex_en, "en"))
check("the FS (raw) rendering is unchanged", idx.render(ex_fa, "en", raw=True) == orig_render(idx, ex_fa, "en", raw=True))

# ---- 4 the frozen capture exemption sees the Persian row in its English form
text = "This kitchen:\n" + rec["text"]["en"] + "\n"
pools = {"practice-k3-v1": set(C.load_manifest()["practice_pool"])}
kw = dict(g1=None, examples=[rec["id"]], items=items, refused=set(), pools=pools, form="package")
try:
    orig_exempt(text, rows=rows, **kw); orig_ok = True
except CP.CaptureAbort:
    orig_ok = False
check("without the mapping the frozen exemption refuses the English line of a Persian row (the capture side of the defect)", orig_ok is False)
spans, audit = CP.exempt_spans(text, rows=rows, **kw)
check("with the mapping it validates the English quote byte for byte against the dataset's English text",
      audit and audit[0]["lang"] == "en" and text[audit[0]["start"]:audit[0]["end"]] == items[IT]["en"])
check("english_row: only Persian rows change, and they record the practised language", FAX.english_row(rows[1], {}) is rows[1]
      and FAX.english_row(rows[0], {IT: "E"}) == dict(rows[0], lang="en", instruction="E", lang_practised="fa"))
bad_row = dict(rows[0], instruction=items["g-l1-pickup-apple"]["fa"])         # an unrelated Persian instruction stored on the row
check("english_row validates the stored Persian instruction against the dataset BEFORE mapping (else unchanged)",
      FAX.english_row(bad_row, {IT: "E"}, {IT: items[IT]["fa"]}) is bad_row)
try:
    CP.exempt_spans(text, rows=[bad_row, rows[1]], **kw); bad_ok = True
except CP.CaptureAbort:
    bad_ok = False
check("a row whose stored Persian instruction is not the dataset text is refused by the frozen exemption (no laundering)", bad_ok is False)
fs_text = "%d. fa | %s | x | ok | success" % (3, items[IT]["fa"])
try:
    CP.exempt_spans(fs_text, rows=rows, **dict(kw, form="fs")); fs_ok = True
except CP.CaptureAbort:
    fs_ok = False
check("the FS form is not mapped (an unchanged Persian FS line still validates as Persian)", fs_ok is True)
FAX.record_event("practice-k3-v1", "practice", fx)
evs = [json.loads(x) for x in open(C.journal_path("practice-k3-v1", "events", fx), encoding="utf-8") if x.strip()]
check("record_event writes the activation boundary with the decision's sha", evs and evs[-1]["event"] == FAX.EVENT
      and evs[-1]["decision_sha256"] == C.sha256_file(FAX.decision_path(fx)))

# ---- 5 wiring
src_p = open(os.path.join(HERE, "k23_practice.py"), encoding="utf-8").read()
src_r = open(os.path.join(HERE, "k23_run.py"), encoding="utf-8").read()
check("k23_practice (live) and k23_run (capture) install it in their own process", "FAX.install()" in src_p and "FAX.install(root)" in src_r)
check("the decision record is bound in the K23 agent freeze", "docs/final-test/k23/decisions/K23-FA-EXAMPLES.json" in
      open(os.path.join(HERE, "k23_freeze_agent.py"), encoding="utf-8").read())
LE.ExampleIndex.render, CP.exempt_spans = orig_render, orig_exempt
shutil.rmtree(fx, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
