"""PRACTICE-V2 gate: reserved acceptance items are excluded from ft-v2 practice (Astra PRACTICE-V2 1-6). Offline, Windows python,
repo root; no simulator, no LLM, no network. Synthetic fixtures only: no reserved item is planned or run.
  1 the reserved set (29 ids, pinned sources, schema) and the reduced practice-v2 pool (112 -> 93; 19 in pool)
  2 the seeded order-0 schedule over the reduced pool (pinned) holds no reserved id; ft-v1 practice is unchanged
  3 fail closed: a reserved draw is refused by the constructor and by the per-attempt guard; changed / missing sources refuse
  4 leakage needles: every reserved id x EN/FA; the snapshot leakage gate catches reserved ids and texts (incl. new slate L3 texts)
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import common as C  # noqa: E402
import practice_k as PK  # noqa: E402

N = {"ok": 0, "bad": 0}


def check(name, cond, detail=""):
    N["ok" if cond else "bad"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


ROOT = C.paths(None)["root"]
with open(C.paths(None)["dataset"], encoding="utf-8") as f:
    GEN = {i["id"]: i for i in json.load(f)["items"]}
MAN = C.load_manifest()

# ---- 1 the reserved set and the pool
R = PK.reserved_acceptance(None, GEN)
check("1.1 29 unique reserved ids", R["n"] == 29 == len(set(R["ids"])))
check("1.2 the reserved-set sha is the PRACTICE-V2 one", R["sha"] == "a183a792c099d34d791738a2ef06144b579e91bf4fbc1baf16e11b292a29cb46", R["sha"])
pr = PK.Practice(namespace="practice-v2", orders=[0], ks=[1, 3, 5, 10], executor=None, channels=(), log=lambda *a: None)
check("1.3 practice-v2 pool 112 -> 93, 19 reserved ids were in the pool",
      (pr.reserved["pool_before"], pr.reserved["pool_after"], len(pr.reserved["in_pool"])) == (112, 93, 19))
check("1.4 no reserved id in the reduced pool", not (set(pr.pool) & set(R["ids"])))
with open(os.path.join(ROOT, "docs", "final-test", "inclusion_manifest_a26.json"), encoding="utf-8") as f:
    ACC = json.load(f).get("acceptance_reserved") or {}         # ACC-ADD (2026-10-14) records the append here (section 7)
if ACC:
    check("1.5 ACC-ADD recorded: the 8 new slate L3 + 2 spares ARE dataset items now (content: section 7)",
          all(i in GEN for i in R["ids"] if i.startswith("g-l3-a2-")))
else:
    check("1.5 the 8 new slate L3 + 2 spares are not dataset items (before ACC-ADD)", all(i not in GEN for i in R["ids"] if i.startswith("g-l3-a2-")))

# ---- 2 the schedule
WANT = ["g-l1-pickup-dish-sponge:fa", "g-l2-microwave-apple:fa", "g-l3-serve-hot-potato:fa", "g-l1-place-mug-counter-top:en",
        "g-l2-find-place-cup-sink-basin:fa", "g-l3-left-open:en", "g-l1-pickup-apple:fa", "g-l2-slice-bread-tomato:en",
        "g-l3-breakfast:fa", "g-l1-pickup-pepper-shaker:fa"]
got = [f"{p['item']}:{p['lang']}" for p in pr.orders_plan[0][:10]]
check("2.1 the order-0 schedule (K <= 10) is the pinned PRACTICE-V2 one", got == WANT, got)
check("2.2 the whole order-0 draw holds no reserved id and no test id",
      not ({p["item"] for p in pr.orders_plan[0]} & (set(R["ids"]) | set(MAN["test"]) | set(MAN["test_uncapped_not_scored"]))))
p1 = PK.Practice(namespace="practice-v1", orders=[0], ks=[0, 1, 3, 5, 10], executor=None, channels=("places",), log=lambda *a: None)
check("2.3 ft-v1 practice untouched: no reserved set, full pool", p1.reserved is None and p1.pool == sorted(MAN["practice_pool"]))
spec = importlib.util.spec_from_file_location("practice_k_pre", os.path.join(ROOT, "sim", "final_test", "_checkpoints", "PRACTICE-V2", "pre",
                                                                             "sim_final_test_matrix_practice_k.py"))
PK0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PK0)
check("2.4 ft-v1 order byte-identical to the pre-change runner",
      PK0.practice_order(p1.pool, PK.level_of_items(GEN), 0) == p1.orders_plan[0])
check("2.5 ft-v1 needles unchanged", p1.needles == PK0.test_needles(MAN, GEN))

# ---- 3 fail closed
orig = PK.practice_order
try:
    PK.practice_order = lambda pool, lv, o: [{"n": 1, "item": "g-l2-find-pickup-mug", "lang": "en", "level": "L2"}]
    check("3.1 a reserved draw is refused before anything runs",
          raises(lambda: PK.Practice(namespace="practice-v2", orders=[0], ks=[1], executor=None, channels=()), PK.ReservedItemRefused))
finally:
    PK.practice_order = orig
check("3.2 per-attempt guard refuses a reserved id", raises(lambda: PK.guard_reserved(["g-l1-pickup-soap-bottle"], R), PK.ReservedItemRefused))
check("3.3 per-attempt guard passes a practice id", PK.guard_reserved(["g-l1-pickup-apple"], R) is None)
src = open(os.path.join(ROOT, "sim", "final_test", "matrix", "practice_k.py"), encoding="utf-8", newline="").read()
check("3.4 attempt() guards every item (ft-v2)", "guard_reserved([item], self.reserved)" in src.split("def attempt(")[1].split("def run_order")[0])
tmp = tempfile.mkdtemp(prefix="pv2_res_")
try:
    for rel, _ in PK.RESERVED_SOURCES.values():
        d = os.path.join(tmp, *rel.split("/")[:-1])
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, *rel.split("/")), os.path.join(tmp, *rel.split("/")))
    check("3.5 byte copies of the sources pass", PK.reserved_acceptance(tmp)["sha"] == R["sha"])
    rel = PK.RESERVED_SOURCES["slate"][0]
    with open(os.path.join(tmp, *rel.split("/")), "ab") as f:
        f.write(b" ")
    check("3.6 a changed slate refuses", raises(lambda: PK.reserved_acceptance(tmp), PK.ReservedItemRefused))
    os.remove(os.path.join(tmp, *rel.split("/")))
    check("3.7 a missing slate refuses", raises(lambda: PK.reserved_acceptance(tmp), PK.ReservedItemRefused))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- 4 leakage needles and gate
have = {(i, lang) for i, lang, _, _ in R["texts"]}
check("4.1 every reserved id has EN and FA needles", all((i, l) in have for i in R["ids"] for l in ("en", "fa")))
check("4.2 practice-v2 needles = test needles + the reserved set",
      set(R["ids"]) <= set(pr.needles["ids"]) and set(PK.test_needles(MAN, GEN)["ids"]) <= set(pr.needles["ids"]))
slate_fa = next(raw for i, lang, _, raw in R["texts"] if i.startswith("g-l3-a2-") and lang == "fa")
res_en = next(raw for i, lang, _, raw in R["texts"] if i == "g-l2-microwave-mug" and lang == "en")
tmp = tempfile.mkdtemp(prefix="pv2_leak_")
try:
    def gate(text):
        d = tempfile.mkdtemp(dir=tmp)
        with open(os.path.join(d, "practice_rows.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"x": text}, ensure_ascii=False))
        return PK.leakage_gate(d, pr.needles)
    check("4.3 a clean snapshot passes", gate("Pick up the apple.")["ok"])
    g = gate("note g-l2-find-pickup-mug")
    check("4.4 a reserved id is a hit", not g["ok"] and any(h["item"] == "g-l2-find-pickup-mug" for h in g["hits"]))
    g = gate(res_en.upper())
    check("4.5 a reserved EN text (normalised) is a hit", not g["ok"] and any(h.get("item") == "g-l2-microwave-mug" for h in g["hits"]))
    g = gate(slate_fa)
    check("4.6 a new slate L3 FA text is a hit", not g["ok"] and any(str(h.get("item")).startswith("g-l3-a2-") for h in g["hits"]))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- 5 field-scoped exemption of the snapshot's own validated instruction field (ft-v2 only; Astra PRACTICE-V2-needles 1-5)
import treatment as T  # noqa: E402
TN = PK.test_needles(MAN, GEN)
test_en = next(raw for i, lang, _, raw in TN["texts"] if i == "g-l1-slice-bread" and lang == "en")
bk_fa = next(raw for i, lang, _, raw in R["texts"] if i == "g-l1-pickup-butter-knife" and lang == "fa")
sched = pr.orders_plan[0][:8]                                  # attempt 8 = g-l2-slice-bread-tomato (en)
EXP = [(p["item"], p["lang"], GEN[p["item"]][p["lang"]]) for p in sched]
REF = set(MAN["test"]) | set(MAN["test_uncapped_not_scored"]) | set(R["ids"])
ALLOW = {"rows_file": T.ROWS_FILE, "expected": EXP, "refused": REF}
it8 = GEN["g-l2-slice-bread-tomato"]
tmp = tempfile.mkdtemp(prefix="pv2_ex_")
try:
    def rows(edit=None, extra_field=None):
        rs = [{"item": i, "lang": l, "instruction": t, "say": "ok"} for i, l, t in EXP]
        if edit:
            edit(rs)
        if extra_field:
            rs[-1]["say"] = extra_field
        return rs

    def snap(rs, other=None):
        d = tempfile.mkdtemp(dir=tmp)
        with open(os.path.join(d, T.ROWS_FILE), "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        if other:
            with open(os.path.join(d, "other.txt"), "w", encoding="utf-8") as f:
                f.write(other)
        return d
    d = snap(rows())
    check("5.1 without the exemption, attempt 8's own instruction is a FALSE hit (the bug)", not PK.leakage_gate(d, pr.needles)["ok"])
    g = PK.leakage_gate(d, pr.needles, allow=ALLOW)
    check("5.2 with it, the same snapshot passes; 8 audited exemptions", g["ok"] and g["exempt"]["n"] == 8 and not g["exempt"]["problems"], g)
    check("5.3 audit entries carry item, lang, field and text sha", all(set(x) == {"item", "lang", "field", "text_sha"} for x in g["exempt"]["exempt"]))
    g = PK.leakage_gate(snap(rows(extra_field=it8["en"])), pr.needles, allow=ALLOW)
    check("5.4 the SAME practice instruction copied into another field (say) still fails", not g["ok"])
    g = PK.leakage_gate(snap(rows(), other=it8["fa"]), pr.needles, allow=ALLOW)
    check("5.5 a practice FA instruction copied into another file still fails", not g["ok"])
    g = PK.leakage_gate(snap(rows(extra_field="tip: " + test_en)), pr.needles, allow=ALLOW)
    check("5.6 a standalone TEST sentence elsewhere fails", not g["ok"] and any(h.get("item") == "g-l1-slice-bread" for h in g["hits"]))
    g = PK.leakage_gate(snap(rows(extra_field=bk_fa)), pr.needles, allow=ALLOW)
    check("5.7 a standalone RESERVED FA sentence elsewhere fails", not g["ok"] and any(h.get("item") == "g-l1-pickup-butter-knife" for h in g["hits"]))
    def swap(rs):
        rs[0]["instruction"] = it8["en"]
    g = PK.leakage_gate(snap(rows(swap)), pr.needles, allow=ALLOW)
    check("5.8 a wrong item / text pair is refused (exemption_invalid)", not g["ok"] and any(h["kind"] == "exemption_invalid" for h in g["hits"]))
    def forge(rs):
        rs[0]["item"] = "g-l1-pickup-cup"
    g = PK.leakage_gate(snap(rows(forge)), pr.needles, allow=ALLOW)
    check("5.9 a forged row (reserved id) is refused and the id is a hit", not g["ok"] and any(h["kind"] == "item_id" for h in g["hits"]))
    def dup(rs):
        rs.append(dict(rs[-1]))
    g = PK.leakage_gate(snap(rows(dup)), pr.needles, allow=ALLOW)
    check("5.10 an unexpected duplicate row is refused", not g["ok"] and any(h["kind"] == "exemption_invalid" for h in g["hits"]))
    d = tempfile.mkdtemp(dir=tmp)
    with open(os.path.join(d, T.ROWS_FILE), "w", encoding="utf-8") as f:
        f.write("not json\n")
    g = PK.leakage_gate(d, pr.needles, allow={"rows_file": T.ROWS_FILE, "expected": EXP[:1], "refused": REF})
    check("5.11 a malformed rows file fails closed", not g["ok"])
    d = tempfile.mkdtemp(dir=tmp)
    g = PK.leakage_gate(d, pr.needles, allow=ALLOW)
    check("5.12 a missing rows file fails closed", not g["ok"])
    upper = rows()
    upper[-1]["instruction"] = it8["en"].upper()
    g = PK.leakage_gate(snap(upper), pr.needles, allow=ALLOW)
    check("5.13 a normalisation variant (upper case) is NOT exempt", not g["ok"])
    g = PK.leakage_gate(snap(rows(extra_field="see g-l2-slice-bread")), pr.needles, allow=ALLOW)
    check("5.16 a whole test id elsewhere still fails (token match)", not g["ok"] and any(h.get("item") == "g-l2-slice-bread" for h in g["hits"]))
    g = PK.leakage_gate(snap(rows(extra_field="G-L2-SLICE-BREAD.")), pr.needles, allow=ALLOW)
    check("5.17 a test id in upper case before punctuation still fails", not g["ok"])
    d = snap(rows())
    check("5.14 allow=None is the old gate (same result as the pre-change runner)", PK0.leakage_gate(d, pr.needles) == PK.leakage_gate(d, pr.needles))
    take = src.split("def take_snapshot(")[1].split("def _request")[0].replace(chr(13) + chr(10), chr(10))
    check("5.15 take_snapshot exempts ft-v2 only, from the PINNED schedule", "if self.ft2:" in take and "self.orders_plan[o][:k]" in take
          and "allow=allow" in take and "leak = leakage_gate(d, self.needles)\n" in take)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- 6 option (A): the G1 request line, recomputed from the practice item's own pinned text (ARASH DECISION 2026-10-13 "Go for A";
#      prereg 6.3 / A24 7(b) amendment DRAFT). Only a byte-identical line in block.text of a validated row is exempt.
take = src.split("def take_snapshot(")[1].split("def _request")[0]
check("6.0 take_snapshot passes the recomputed request line (4th element) for ft-v2", "request_line(self.items[p[\"item\"]], p[\"lang\"])" in take)
sched10 = pr.orders_plan[0][:10]
RL = {p["item"]: PK.request_line(GEN[p["item"]], p["lang"]) for p in sched10}
check("6.1 recompute: attempt 2 (fa) renders the G1 line with the embedded test sentence",
      RL["g-l2-microwave-apple"] == "The request in English: put Apple to the microwave; close the microwave.", RL["g-l2-microwave-apple"])
check("6.2 recompute: the 3 L3 items render no G1 line", all(RL[i] is None for i in ("g-l3-serve-hot-potato", "g-l3-left-open", "g-l3-breakfast")))
JR = os.path.join(ROOT, "docs", "final-test", "matrix", "practice-v2", "practice.jsonl")
live = []
if os.path.exists(JR):
    with open(JR, encoding="utf-8") as f:
        live = [json.loads(ln) for ln in f if ln.strip()][:2]
EXP4 = [(p["item"], p["lang"], GEN[p["item"]][p["lang"]], RL[p["item"]]) for p in sched10]
A4 = lambda k: {"rows_file": T.ROWS_FILE, "expected": EXP4[:k], "refused": REF}
A3 = lambda k: {"rows_file": T.ROWS_FILE, "expected": [e[:3] for e in EXP4[:k]], "refused": REF}
tmp = tempfile.mkdtemp(prefix="pv2_rq_")
try:
    def snap2(rs, other=None):
        d = tempfile.mkdtemp(dir=tmp)
        with open(os.path.join(d, T.ROWS_FILE), "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        if other:
            with open(os.path.join(d, "other.txt"), "w", encoding="utf-8") as f:
                f.write(other)
        return d
    if len(live) == 2:
        lines2 = live[1]["block"]["text"].split("\n")
        check("6.3 the LIVE row 2 request line is byte-identical to the recompute", RL["g-l2-microwave-apple"] in lines2)
        g = PK.leakage_gate(snap2(live), pr.needles, allow=A3(2))
        check("6.4 live rows 1-2 WITHOUT option (A): the literal gate hits the test id g-l1-close-microwave (the blocker)",
              not g["ok"] and [h.get("item") for h in g["hits"]] == ["g-l1-close-microwave"], g["hits"])
        g = PK.leakage_gate(snap2(live), pr.needles, allow=A4(2))
        check("6.5 live rows 1-2 WITH option (A): pass, 2 instruction + 2 request-line exemptions audited",
              g["ok"] and g["exempt"]["n"] == 2 and g["exempt"]["reqline_n"] == 2, g)
        check("6.6 request-line audit carries item, lang, field, text sha and count",
              all(set(x) == {"item", "lang", "field", "text_sha", "count"} and x["field"].endswith(".block.text#request_line")
                  for x in g["exempt"]["reqline"]))
    else:
        check("6.3-6.6 the live practice-v2 journal (rows 1-2) is present", False, JR)

    def syn(k):                                          # synthetic rows for the first k scheduled presentations
        rs = []
        for i, l, t, rq in EXP4[:k]:
            body = ["Steps that work here:"] + ([rq] if rq else []) + ["close X: closes it."]
            rs.append({"item": i, "lang": l, "instruction": t, "say": "ok", "block": {"text": "\n".join(body), "n_records": 2}})
        return rs
    g = PK.leakage_gate(snap2(syn(10)), pr.needles, allow=A4(10))
    check("6.7 synthetic K=10 rows with every recomputed line pass (7 request lines, 10 instruction exemptions)",
          g["ok"] and g["exempt"]["reqline_n"] == 7 and g["exempt"]["n"] == 10, g)
    g = PK.leakage_gate(snap2(syn(10)), pr.needles, allow=A3(10))
    check("6.8 the same rows under the instruction-only exemption still fail (the request line is not silently exempt)", not g["ok"])

    def mut(fn, k=3):
        rs = syn(k)
        fn(rs)
        return PK.leakage_gate(snap2(rs), pr.needles, allow=A4(k))
    inv = lambda g: (not g["ok"]) and any(h["kind"] == "exemption_invalid" for h in g["hits"])
    leak_t = lambda g, i: (not g["ok"]) and any(h.get("item") == i for h in g["hits"])
    def b(rs, k, s):
        rs[k]["block"]["text"] = s
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"].replace("microwave.", "microwave!")))
    check("6.9 a request line differing by one character is refused", inv(g), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"].replace("microwave.", "microwave. ")))
    check("6.10 a trailing space is refused (byte identity)", inv(g), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"] + "\n" + RL["g-l2-microwave-apple"]))
    check("6.11 a duplicated request line in the same block is refused", inv(g), g["hits"])
    g = mut(lambda rs: b(rs, 0, rs[0]["block"]["text"] + "\n" + RL["g-l2-microwave-apple"]))
    check("6.12 row 2's request line placed in row 1's block is refused (wrong row / item)", inv(g), g["hits"])
    g = mut(lambda rs: b(rs, 2, rs[2]["block"]["text"] + "\n" + RL["g-l2-microwave-apple"]))
    check("6.13 a request line in an L3 row whose recompute is None is refused", inv(g), g["hits"])
    g = mut(lambda rs: rs[1].update(say=RL["g-l2-microwave-apple"]))
    check("6.14 the exact request line copied into another field (say) still leaks", leak_t(g, "g-l1-close-microwave"), g["hits"])
    rs = syn(3)
    g = PK.leakage_gate(snap2(rs, other=RL["g-l2-microwave-apple"]), pr.needles, allow=A4(3))
    check("6.15 the exact request line copied into another file still leaks", leak_t(g, "g-l1-close-microwave"), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"] + "\nTip: " + test_en))
    check("6.16 a test sentence on ANOTHER line of the same block still leaks", leak_t(g, "g-l1-slice-bread"), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"] + "\n" + bk_fa))
    check("6.17 a reserved FA sentence on another line of the same block still leaks", leak_t(g, "g-l1-pickup-butter-knife"), g["hits"])
    g = mut(lambda rs: rs[1].update(lang="en"))
    check("6.18 a row whose lang differs from the schedule gets no exemption at all", inv(g) and leak_t(g, "g-l1-close-microwave"), g["hits"])
    def fake(rs):
        rs[1]["instruction"] = "Put the apple in the microwave and close the microwave."
    g = mut(fake)
    check("6.19 a row whose instruction is not the pinned text gets no request-line exemption", inv(g) and leak_t(g, "g-l1-close-microwave"), g["hits"])
    g = mut(lambda rs: rs[1].update(block={"text": 7}))
    check("6.20 a non-string block.text fails closed", inv(g), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"].replace("The request in English: ", "the request in english: ")))
    check("6.21 a case variant of the line is not exempt (it still leaks)", leak_t(g, "g-l1-close-microwave"), g["hits"])
    g = mut(lambda rs: b(rs, 1, rs[1]["block"]["text"] + "\nsee g-l2-slice-bread"))
    check("6.22 ids stay unconditional inside an exempted block", leak_t(g, "g-l2-slice-bread"), g["hits"])
    real = PK.request_line
    try:
        PK.request_line = lambda item, lang: "The request in English: close the microwave."
        check("6.23 the gate never calls the recompute itself (the caller pins it; the snapshot cannot authorise itself)",
              PK.leakage_gate(snap2(syn(3)), pr.needles, allow=A4(3))["ok"])
    finally:
        PK.request_line = real
    d = snap2(syn(3))
    check("6.24 allow=None (ft-v1 gate) on rows with request lines equals the pre-change runner's gate",
          PK.leakage_gate(d, pr.needles) == PK0.leakage_gate(d, pr.needles))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- 7 ACC-ADD (2026-10-14; Arash's card-14 approval): the 10 approved new L3 acceptance items in general.json. Runs only once the
#      inclusion manifest records the append (before it, 1.5 asserts their absence). Reserved = in no split group + the pinned slate.
NEW = [i for i in R["ids"] if i.startswith("g-l3-a2-")]
if ACC:
    with open(os.path.join(ROOT, *PK.RESERVED_SOURCES["slate"][0].split("/")), encoding="utf-8") as f:
        SL = json.load(f)
    SLI = [x for x in SL["l3_new"] + SL["l3_spares"]]
    check("7.1 the recorded ids = the slate's 8 main + 2 spares, in order, = the 10 reserved g-l3-a2 ids",
          ACC.get("items") == [x["id"] for x in SLI] and sorted(ACC["items"]) == sorted(NEW) and len(NEW) == 10, ACC.get("items"))
    check("7.2 each stored item = the approved slate EN / FA / setup / plan / goal exactly, level L3, reviewed true, dataset shape",
          all(GEN[x["id"]] == {"id": x["id"], "level": "L3", "en": x["en"], "fa": x["fa"], "setup": x["setup"], "plan": x["plan"],
                               "goal": x["goal"], "reviewed": True} and list(GEN[x["id"]]) == ["id", "level", "en", "fa", "setup", "plan",
                                                                                         "goal", "reviewed"] for x in SLI))
    check("7.3 the effective manifest is pinned to the live dataset (A26 successor style re-pin)",
          (MAN.get("inclusion") or {}).get("dataset_sha256") == C.sha256_file(C.paths(None)["dataset"])
          and "the gold dataset differs from the one the split was cut from" not in C.dataset_problems("test-v1"))
    groups = {k: set(v) for k, v in MAN.items() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)}
    check("7.4 in NO split group of the effective manifest", not any(set(NEW) & g for g in groups.values()),
          {k: sorted(set(NEW) & g) for k, g in groups.items() if set(NEW) & g})
    split_ns = [ns for ns in C.NAMESPACES if C.NAMESPACES[ns]["items"] != getattr(C, "ACCEPTANCE_ROSTER", None)]
    refused = all(raises(lambda ns=ns, i=i: C.guard_items([i], ns), C.TestItemRefused) for ns in split_ns for i in NEW)
    acc_ok = True                                                        # ACC-RUNNER: acceptance-v2 admits exactly its registered roster
    if len(split_ns) < len(C.NAMESPACES):
        import acc_common as _AC
        _ro = set(_AC.acceptance_roster())
        acc_ok = all(raises(lambda i=i: C.guard_items([i], C.ACCEPTANCE_NS), C.TestItemRefused) for i in NEW)   # no session: all refused
        _AC.SESSION.update(armed=True, by="selftest")
        try:
            acc_ok = acc_ok and all(raises(lambda i=i: C.guard_items([i], C.ACCEPTANCE_NS), C.TestItemRefused) != (i in _ro) for i in NEW)
        finally:
            _AC.disarm_session()
    check("7.5 the split guard refuses every one in every split namespace (%d); acceptance-v2 admits exactly its registered roster"
          % len(split_ns), refused and acc_ok)
    check("7.6 practice pools unchanged: practice-v2 93 (reserved sha as pinned), ft-v1 = the manifest pool, neither holds a new id",
          len(pr.pool) == 93 and R["sha"] == "a183a792c099d34d791738a2ef06144b579e91bf4fbc1baf16e11b292a29cb46"
          and not (set(NEW) & (set(pr.pool) | set(p1.pool))))
    import ltm_package as LPK  # noqa: E402
    import ltm_templates as LTP  # noqa: E402
    check("7.7 the manifest-driven builders do not see them (ltm_package.nontest_items, ltm_templates.load_nontest_items)",
          not ({x["id"] for x in LPK.nontest_items()} & set(NEW)) and not ({x["id"] for x in LTP.load_nontest_items()} & set(NEW)))
    check("7.8 ft-v2 practice needles name every one (id + EN/FA text)", set(NEW) <= set(pr.needles["ids"])
          and all((i, l) in {(t[0], t[1]) for t in pr.needles["texts"]} for i in NEW for l in ("en", "fa")))
    with open(os.path.join(ROOT, "sim", "dataset", "general_audit.jsonl"), encoding="utf-8") as f:
        AUD = [json.loads(ln) for ln in f if ln.strip()]
    adds = {a["id"]: a for a in AUD if a.get("op") == "add" and a.get("id") in NEW}
    check("7.9 the audit log holds one append-only 'add' per item (before null, after = the stored item)",
          len([a for a in AUD if a.get("op") == "add" and a.get("id") in NEW]) == 10
          and all(adds[i]["before"] is None and adds[i]["after"] == GEN[i] for i in NEW))

print(f"_ftm_practice_reserved_selftest: {N['ok']}/{N['ok'] + N['bad']} passed")
sys.exit(1 if N["bad"] else 0)
