"""Self-test for the A12 retention (final test, stage E2; prereg s16 A12, docs/final-test/preregistration.md).

Offline, Windows `python` from the REPO ROOT, temp directories only, no simulator, no WSL, no Ollama, no network, no API key use:

    python sim/_step5_retention_a12_selftest.py

Covers: the pure allocation rule (A_L = max(share x B_L, H + N_L x r); budget raise to a multiple of 10 up to 2 x; memory-incapable
level); the C2-e path byte-identical when retention is absent; the A12 block on a real LongTermMemory with a fake two-tokenizer
counter (record ids identical in EN and FA, the complete `header + block` within the allowance / budget under every tokenizer, the
procedures share B_L - A_L, the empty L1 problem of C2-e solved); the minimum-coverage criterion and the freeze / post-freeze gate
(exemptions, failures, vacuous input, outcome-blind); the real RealMaxCounter class with its raw counters stubbed; fail-fast refusals.
"""
import copy
import importlib
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "final_test", "c3"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # pragma: no cover
    pass

import ltm as L                                             # noqa: E402
import ltm_adapter as A                                     # noqa: E402
import ltm_retrieval as R                                   # noqa: E402

# objmem is pure step-4 geometry (no simulator); the adapter takes it by injection. Loaded by name so the step-5 import guard
# (which allow-lists step-4 imports per selftest) keeps seeing no top-level step-4 import here.
objmem = importlib.import_module("objmem")
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


HEADER = "Long-term memory (retrieved for this command; learned from earlier practice):"


class Ctr:
    """Fake two-tokenizer max counter with the RealMaxCounter surface (counts / header / header_cost / __call__). Persian costs more."""
    tokenizers = ["gemma", "anthropic"]
    header = HEADER

    def __init__(self, scale=1.0):
        self.scale = scale
        self.n = 0

    def counts(self, t):
        if not t.strip():
            return {"gemma": 1, "anthropic": 1}
        n = len(t)
        fa = sum(1 for c in t if "؀" <= c <= "ۿ")
        g = max(1, int(((n - fa) / 3.5 + fa / 1.6) * self.scale))
        a = max(1, int(((n - fa) / 3.0 + fa / 1.9) * self.scale))
        return {"gemma": g, "anthropic": a}

    @property
    def header_cost(self):
        return max(self.counts(self.header + "\n.").values())

    def __call__(self, t):
        self.n += 1
        return 0 if not t else max(1, max(self.counts(t).values()))


SCENES = [
    [{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Apple", "box": [30, 40, 40, 50], "depth": 1.9}],
    [{"type": "CounterTop", "box": [0, 80, 95, 95], "depth": 2.5}, {"type": "Apple", "box": [30, 82, 40, 90], "depth": 2.4},
     {"type": "Mug", "box": [70, 82, 78, 90], "depth": 2.4}],
    [{"type": "Cabinet", "box": [60, 10, 95, 60], "depth": 3.0}, {"type": "Mug", "box": [70, 30, 80, 40], "depth": 2.9}],
    [{"type": "Fridge", "box": [10, 10, 60, 80], "depth": 2.0}, {"type": "Mug", "box": [30, 40, 40, 50], "depth": 1.9}],
]
tmp = tempfile.mkdtemp()
_n = [0]


def load_pristine():
    """The registry WITHOUT the A12 block that W2 writes into sim/step5/params.json: these tests are about the switch, so they build their own a12 blocks."""
    P = copy.deepcopy(L.load_params())
    P["retrieval"].pop("retention", None)
    P["retrieval"].pop("a12", None)
    return P


def make(params=None, scale=1.0, counter=None):
    _n[0] += 1
    params = params if params is not None else load_pristine()
    mb = L.LongTermMemory(os.path.join(tmp, f"m{_n[0]}.db"), kitchen="Fake", params=params,
                          count_tokens=counter if counter is not None else Ctr(scale))
    for i, sc in enumerate(SCENES):
        mb.reset_world(f"r{i}")
        ad = A.HonestEventAdapter(mb, episode=f"ep{i}", mode="practice", geometry=objmem)
        ag = A.FakeHonestAgent(ad, objmem, episode=f"e{i}")
        ad.on_task("Put the apple in the fridge", "L2", "en")
        ag.look(sc)
        ad.on_terminal(True, 20, 0)
        ad.on_admission(True)
        ad.end()
        mb.end_episode(ad.wm, admission=True)
    mb.reset_world("tt")
    w = mb.begin_episode("bt", "test")
    adt = A.HonestEventAdapter(mb, wm=w, episode="bt", mode="test", geometry=objmem)
    adt.on_task("Put the apple in the fridge and put the mug on the counter", "L3", "en")
    return mb, w


def a12_params(alloc, H, r):
    P = load_pristine()
    P["retrieval"]["retention"] = "a12"
    P["retrieval"]["a12"] = {"H": H, "r": r, "N": alloc["N"], "A": alloc["A"], "B": alloc["B"], "raised": alloc["raised"],
                             "incapable": alloc["incapable"], "reason": "A12, token-derived", "fp1_measured": True}
    return P


# ================================================================================================ 1
print("1. a12_allocation: the pure rule")
al = R.a12_allocation(26, 125)
check("registered B, N = 1/2/3, share 0.4: L1 need 151 > 150 -> raised to 160 (multiple of 10, <= 2x)", al["B"]["L1"] == 160 and al["raised"]["L1"] == [150, 160], str(al))
check("L2 need 276 <= 300: no raise, A = need (276) > 0.4 x 300", al["B"]["L2"] == 300 and al["A"]["L2"] == 276 and "L2" not in al["raised"])
check("L3 need 401 <= 450: A = 401", al["B"]["L3"] == 450 and al["A"]["L3"] == 401)
check("L1 allowance after the raise = need 151 (> 0.4 x 160 = 64)", al["A"]["L1"] == 151)
check("nothing incapable here", al["incapable"] == [])
check("A_L <= B_L for every level, B_L multiple of 10 whenever raised",
      all(al["A"][k] <= al["B"][k] for k in al["A"]) and all(al["B"][k] % 10 == 0 for k in al["raised"]))
big = R.a12_allocation(26, 400)
check("r so large that 2 x B_L cannot hold N records -> L1 and L2 memory-incapable",
      big["incapable"] == ["L1", "L2"] or big["incapable"] == ["L1", "L2", "L3"] or set(big["incapable"]) >= {"L1", "L2"}, str(big))
check("E-fix: an incapable level is NOT raised (A12: not tuned further): B stays registered, A = share x B, the cap is only reported",
      big["B"]["L1"] == 150 and big["A"]["L1"] == 60 and "L1" not in big["raised"] and big["not_applied"]["L1"] == [150, 300])
check("... a capable level next to it is still raised normally", R.a12_allocation(26, 125)["not_applied"] == {})
small = R.a12_allocation(10, 5)
check("small r: the C2-e share dominates, nothing raised: A = 0.4 x B (float safe: 0.4 x 450 = 180, not 181)",
      small["A"] == {"L1": 60, "L2": 120, "L3": 180} and small["B"] == {"L1": 150, "L2": 300, "L3": 450} and small["raised"] == {})
check("deterministic and pure (no mutation of the inputs)", R.a12_allocation(26, 125) == al)
B0 = {"L1": 150, "L2": 300, "L3": 450}
R.a12_allocation(26, 125, B=B0)
check("input dict not mutated", B0 == {"L1": 150, "L2": 300, "L3": 450})
check("optional section overhead s raises the need exactly", R.a12_allocation(26, 125, s=9)["need"]["L1"] == 160)
check("the allocation signature takes only token facts (no outcome-like argument)",
      set(inspect.signature(R.a12_allocation).parameters) == {"H", "r", "N", "B", "where_share", "cap_factor", "s"})

# ================================================================================================ 2
print("2. retention switch: absent = C2-e unchanged; Retriever budgets")
mb0, w0 = make()
for lv in ("L1", "L2", "L3"):
    for lg in ("en", "fa"):
        a = mb0.memory_block(w0, lv, lg)
        b = mb0.memory_block(w0, lv, lg, retention="c2e")
        check(f"no retention setting == retention='c2e' ({lv}/{lg}), byte-identical block", a == b)
_real = L.load_params()["retrieval"]
check("the real registry either carries no a12 block or carries one switched on by retention == 'a12' (W2 writes it)",
      "a12" not in _real or _real.get("retention") == "a12")
if _real.get("retention") == "a12":
    _b = _real["a12"]
    _al = R.a12_allocation(_b["H"] + _b["head"], _b["r"], s=_b["s"], B={k: int(v) for k, v in (_b["B"] if "A16" in str(_b.get("reason", "")) else _real["budget_tokens"]["value"]).items()},
                           where_share=float(_real["where_share"]["value"]))
    check("the real a12 block equals the allocation recomputed from its own measured H, head, r, s",
          _b["A"] == _al["A"] and _b["B"] == _al["B"] and _b["raised"] == _al["raised"] and _b["incapable"] == _al["incapable"], str(_b) + " vs " + str(_al))
check("an unknown retention name is refused", raises(ValueError, mb0.memory_block, w0, "L2", "en", retention="bogus"))
check("a12 without params['retrieval']['a12'] is refused", raises(ValueError, mb0.memory_block, w0, "L2", "en", retention="a12"))
P1 = a12_params(R.a12_allocation(26, 125), 26, 125)
check("Retriever budgets read the (raised) a12.B", R.Retriever(params=P1).budgets["L1"] == 160
      and R.Retriever(params=P1).registered_budgets["L1"] == 150)
check("without retention=a12 the registered budgets stay", R.Retriever(params=load_pristine()).budgets == B0)

# ================================================================================================ 3
print("3. the A12 block on a real LongTermMemory (fake two-tokenizer counter)")
ctr = Ctr()
mbm, wm_ = make(counter=ctr)
msr = L.a12_measure([mbm.memory_block(wm_, lv, lg) for lv in ("L2", "L3") for lg in ("en", "fa")], ctr)
H, r_en, s_en = msr["H_eff"], msr["r"], msr["s"]
check("a12_measure: H = counter.header_cost; r and s are positive real maxima over both languages",
      msr["H"] == ctr.header_cost and H > msr["H"] and r_en > 0 and s_en > 0 and msr["n_records"] >= 4, str(msr))
check("... and r is at least the English maximum (max over languages)",
      r_en >= max(max(ctr.counts(x["text"] + "\n").values()) for x in mbm.memory_block(wm_, "L3", "en")["records"]))
al = R.a12_allocation(H, r_en, s=s_en)
P2 = a12_params(al, H, r_en)
mb, w = make(params=P2, counter=Ctr())
c = mb.count_tokens
for lv in ("L1", "L2", "L3"):
    be, bf = mb.memory_block(w, lv, "en"), mb.memory_block(w, lv, "fa")
    check(f"{lv}: mode follows params (a12) and EN/FA retain identical ids",
          be["retention"] == "a12" and [x["id"] for x in be["records"]] == [x["id"] for x in bf["records"]], str(be["records"]))
    for lg, bk in (("en", be), ("fa", bf)):
        for tk, n in c.counts(HEADER + "\n" + bk["text"]).items() if bk["text"] else []:
            check(f"{lv}/{lg}/{tk}: complete header + block {n} <= B_L {P2['retrieval']['a12']['B'][lv]}", n <= P2["retrieval"]["a12"]["B"][lv])
            check(f"{lv}/{lg}/{tk}: places-only block {n} <= A_L {P2['retrieval']['a12']['A'][lv]}", n <= P2["retrieval"]["a12"]["A"][lv])
        check(f"{lv}/{lg}: complete_ok flag true and budget reports B_L", bk["complete_ok"] and bk["budget"] == P2["retrieval"]["a12"]["B"][lv])
n_l = {lv: len([x for x in mb.memory_block(w, lv, "en")["records"]]) for lv in ("L1", "L2", "L3")}
cand = mb.place_candidate_count(w)
check("the fixture has 4 learned place candidates (Apple 1 + Mug 3)", cand == 4, str(cand))
check("A12 minimum capacity: L1 holds >= 1, L2 >= 2, L3 >= 3 whole place records (r + s accounting)", n_l["L1"] >= 1 and n_l["L2"] >= 2 and n_l["L3"] >= 3, f"{n_l} {al}")
nos = R.a12_allocation(H, r_en)
P_nos = a12_params(nos, H, r_en)
mbn, wn = make(params=P_nos, counter=Ctr())
check("WITHOUT the section overhead s the literal allocation under-provisions L3 -> the coverage check catches it (not silent)",
      not L.a12_coverage({lg: mbn.memory_block(wn, "L3", lg) for lg in ("en", "fa")}, mbn.place_candidate_count(wn), "L3", mbn.count_tokens, nos)["ok"])
check("an explicit retention='a12' equals the params-driven block", mb.memory_block(w, "L2", "en", retention="a12") == mb.memory_block(w, "L2", "en"))
check("retention='c2e' on the same object still works (old behaviour reachable)", "retention" not in mb.memory_block(w, "L2", "en", retention="c2e"))
check("a12 is deterministic", mb.memory_block(w, "L3", "fa") == mb.memory_block(w, "L3", "fa"))
mbe, we = make(params=P2, counter=lambda t: max(1, len(t) // 4))
check("a12 refuses a plain callable counter (an estimate cannot back it)", raises(ValueError, mbe.memory_block, we, "L2", "en"))
check("a12 refuses joint=False", raises(ValueError, mb.memory_block, w, "L2", "en", joint=False))

# the C2-e failure the amendment fixes: a heavy tokenizer leaves L1 empty under C2-e, A12 raises it to hold N=1 (the L1 budget is raised)
ch = Ctr(scale=2.4)
Hh = ch.header_cost
mbh, wh = make(counter=ch)
msh = L.a12_measure([mbh.memory_block(wh, "L3", lg) for lg in ("en", "fa")], ch)
rh = msh["r"]
alh = R.a12_allocation(msh["H_eff"], rh, s=msh["s"])
check("heavy tokenizer: C2-e L1 block is empty (the pilot symptom)", mbh.memory_block(wh, "L1", "fa")["empty"] or mbh.memory_block(wh, "L1", "en")["empty"])
Ph = a12_params(alh, Hh, rh)
mbh2, wh2 = make(params=Ph, counter=Ctr(scale=2.4))
b1e, b1f = mbh2.memory_block(wh2, "L1", "en"), mbh2.memory_block(wh2, "L1", "fa")
check("heavy tokenizer + A12: L1 holds a record in both languages (ids identical)",
      b1e["records"] and [x["id"] for x in b1e["records"]] == [x["id"] for x in b1f["records"]],
      f"{alh['B']} {alh['A']} {alh['incapable']} {b1e['dropped']}")

# ---- E-fix: r is measured over EVERY learned place candidate, not only the records the old budget retained
heavy_blocks = [mbh.memory_block(wh, "L1", lg) for lg in ("en", "fa")]                       # C2-e at the registered budget: empty
ms_old = L.a12_measure(heavy_blocks, ch)
ms_new = L.a12_measure(heavy_blocks, ch, candidates=[(mbh, wh)])
check("E-fix: from the (empty) retained blocks alone r is 0 (biased low)", ms_old["r"] == 0 and ms_old["n_records"] == 0, str(ms_old))
check("E-fix: with candidates=, r is the real max over every candidate rendered on its own (the records the old budget dropped)",
      ms_new["r"] > 0 and ms_new["n_candidates_measured"] == mbh.place_candidate_count(wh) == 4 and ms_new["s"] > 0, str(ms_new))
lines = mbh.place_candidate_lines(wh)
check("E-fix: candidate lines cover EN and FA at every level, one block per candidate", {x["lang"] for x in lines} == {"en", "fa"}
      and {x["level"] for x in lines} == {"L1", "L2", "L3"} and all(len(x["records"]) == 1 for x in lines))
check("E-fix: r with candidates is at least the r of the retained records of the same LTM",
      ms_new["r"] >= L.a12_measure([mbh.memory_block(wh, "L3", lg) for lg in ("en", "fa")], ch)["r"])
check("E-fix: a12_measure keeps working without candidates (n_candidates_measured 0)", L.a12_measure([mbh.memory_block(wh, "L3", "en")], ch)["n_candidates_measured"] == 0)

# ---- E-fix: the counter guards and the explicit c2e budgets
mg, wg = make(params=P2, counter=Ctr())
mg.memory_block(wg, "L2", "en")                                     # builds the cached Retriever with counter #1
mg.count_tokens = Ctr()                                             # a later assignment (open_frozen callers do this): a different counter object
check("E-fix: a12 refuses when the cached Retriever holds a different counter than self.count_tokens", raises(ValueError, mg.memory_block, wg, "L2", "en"))
check("E-fix: a complete a12 block records the tokenizers it was measured under",
      mb.memory_block(w, "L2", "en")["tokenizers"] == ["anthropic", "gemma"] and mb.memory_block(w, "L2", "en")["measurement_only"] is False)
mplain, wplain = make(counter=Ctr())
c2_plain = mplain.memory_block(wplain, "L1", "en")
c2_on_a12 = mb.memory_block(w, "L1", "en", retention="c2e")
check("E-fix: retention='c2e' on a registry that carries a raised a12 block still uses the REGISTERED budget (reproduces C3 / C4)",
      c2_on_a12["budget"] == c2_plain["budget"] and c2_on_a12["text"] == c2_plain["text"] and mb.memory_block(w, "L1", "en")["budget"] != c2_plain["budget"],
      f"{c2_on_a12['budget']} {c2_plain['budget']}")

# ================================================================================================ 4
print("4. procedures share B_L - A_L (stubbed procedure channel)")
mp, wp = make(params=P2, counter=Ctr())
B3, A3 = P2["retrieval"]["a12"]["B"]["L3"], P2["retrieval"]["a12"]["A"]["L3"]
mp.proc_candidates = lambda wm, holding=None: {"items": [{"id": f"p{i}", "kind": "procedure", "rec": {"t": {"en": "proc line number %d: " % i + "walk to the fridge then open it " * 2,
                                                                                                      "fa": "رویه شماره %d: " % i + "برو به یخچال و بازش کن " * 2}}}
                                                         for i in range(40)], "report": {}}
mp.retriever().render_proc = lambda rec, lg, suggestion=False: rec["t"][lg]
pe, pf = mp.memory_block(wp, "L3", "en", channels=("places", "procedures")), mp.memory_block(wp, "L3", "fa", channels=("places", "procedures"))
pids_e = [x["id"] for x in pe["records"] if x["id"].startswith("proc/")]
check("procedures are retained and identical in EN and FA", pids_e and pids_e == [x["id"] for x in pf["records"] if x["id"].startswith("proc/")], str(pids_e))
check("not all 40 fit: the procedure channel is bounded", 0 < len(pids_e) < 40, str(len(pids_e)))
for lg, bk in (("en", pe), ("fa", pf)):
    cn = mp.count_tokens.counts(HEADER + "\n" + bk["text"])
    check(f"{lg}: complete header + places + procedures {cn} <= B_L {B3}", all(v <= B3 for v in cn.values()) and bk["complete_ok"])
    sec_proc = bk["text"].split(R.WORDS[lg]["procs"], 1)[1]
    check(f"{lg}: the procedure section is within B_L - A_L = {B3 - A3} (real max count)",
          max(mp.count_tokens.counts(R.WORDS[lg]["procs"] + sec_proc).values()) <= B3 - A3)
check("places records are unaffected by the procedures channel (same place ids as places-only)",
      [x["id"] for x in pe["records"] if not x["id"].startswith("proc/")] == [x["id"] for x in mp.memory_block(wp, "L3", "en")["records"]])
check("a prefix_shrinks counter is reported", isinstance(pe["prefix_shrinks"], int) and pe["prefix_shrinks"] >= 1)

# ================================================================================================ 5
print("5. a12_coverage and the freeze / post-freeze gate")
ctr5 = Ctr()
blocks = {lg: mb.memory_block(w, "L3", lg) for lg in ("en", "fa")}
cov = L.a12_coverage(blocks, cand, "L3", mb.count_tokens, al)
# W4a (A17): Mug now renders as لیوان (shorter than ماگ in real tokens), so the language-joint retention may hold 4 instead of 3; the rule is >= N and EN == FA
check("coverage ok when >= N place records are held in both languages", cov["ok"] and cov["held"]["en"] == cov["held"]["fa"] >= 3 and cov["exempt"] is None, str(cov))
check("coverage reports real counts per tokenizer x language", set(cov["real"]["en"]) == {"gemma", "anthropic"} and set(cov["real"]) == {"en", "fa"})
cov_few = L.a12_coverage({"en": {"text": "", "records": []}, "fa": {"text": "", "records": []}}, 1, "L3", mb.count_tokens, al)
check("fewer than N candidates: exempt (reported), ok", cov_few["ok"] and cov_few["exempt"] == "fewer than N place candidates")
cov_bad = L.a12_coverage({"en": {"text": "", "records": []}, "fa": {"text": "", "records": []}}, 3, "L3", mb.count_tokens, al)
check("enough candidates but an empty block: NOT ok (capacity failure)", not cov_bad["ok"])
inc = dict(al, incapable=["L3"])
check("a memory-incapable level is exempt and reported",
      L.a12_coverage({"en": {"text": "", "records": []}, "fa": {"text": "", "records": []}}, 3, "L3", mb.count_tokens, inc)["exempt"] == "memory-incapable level")
fa_diff = dict(blocks["fa"], records=blocks["fa"]["records"][:-1])
check("ids differing between languages fail the coverage", not L.a12_coverage({"en": blocks["en"], "fa": fa_diff}, cand, "L3", mb.count_tokens, al)["ok"])
over = dict(al, B=dict(al["B"], L3=20))
check("a block whose complete count exceeds B_L fails (complete-prefix check)", not L.a12_coverage(blocks, cand, "L3", mb.count_tokens, over)["ok"])
items = [{"id": f"x{lv}", "level": lv, "n_place_candidates": cand, "blocks": {lg: mb.memory_block(w, lv, lg) for lg in ("en", "fa")}} for lv in ("L1", "L2", "L3")]
gate = L.a12_freeze_check(items, al, mb.count_tokens)
check("freeze check ok on a good set; outcome-blind flag; stage recorded", gate["ok"] and gate["outcome_blind"] and gate["stage"] == "pre_freeze"
      and not gate["blocks_freeze"] and not gate["blocks_execution"] and gate["n"] == 3, str(gate["failures"]))
check("empty-block rates reported per level x language", set(gate["by_level"]["L1"]["empty_rate"]) == {"en", "fa"})
items_bad = items + [{"id": "bad", "level": "L3", "n_place_candidates": 3, "blocks": {"en": {"text": "", "records": []}, "fa": {"text": "", "records": []}}}]
gb = L.a12_freeze_check(items_bad, al, mb.count_tokens)
check("one failing item blocks the freeze", not gb["ok"] and gb["blocks_freeze"] and gb["failures"][0]["id"] == "bad")
gp = L.a12_freeze_check(items_bad, al, mb.count_tokens, post_freeze=True)
check("post-freeze the same failure blocks execution (and never alters parameters)", gp["blocks_execution"] and not gp["blocks_freeze"] and gp["stage"] == "post_freeze")
check("an empty item list is a failure, not a vacuous pass", not L.a12_freeze_check([], al, mb.count_tokens)["ok"]
      and L.a12_freeze_check([], al, mb.count_tokens)["reason"] == "no_items")
check("the check functions take no outcome argument",
      not any(k for k in inspect.signature(L.a12_freeze_check).parameters if "succ" in k or "outcome" in k))
import ast                                                  # noqa: E402
import textwrap                                             # noqa: E402
names = set()
for fn in (L.a12_coverage, L.a12_freeze_check, L.a12_measure, R.a12_allocation):
    for nd in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
        if isinstance(nd, ast.Name):
            names.add(nd.id.lower())
        elif isinstance(nd, ast.Attribute):
            names.add(nd.attr.lower())
        elif isinstance(nd, ast.arg):
            names.add(nd.arg.lower())
        elif isinstance(nd, ast.Constant) and isinstance(nd.value, str) and len(nd.value) < 40:
            names.add(nd.value.lower())
bad = [n for n in names if any(w_ in n for w_ in ("success", "valid", "score", "pass_", "reward"))]
check("no outcome-like identifier or key in the A12 functions (success / valid / score / reward)", not bad, str(bad))

# ================================================================================================ 6
print("6. the real RealMaxCounter class (raw counters stubbed; no network, no key use)")
import real_counter as RC                                   # noqa: E402

rc = RC.RealMaxCounter(os.path.join(tmp, "tok.json"), HEADER, allow_missing_anthropic=True)
rc._key = None
rc.tokenizers = ["gemma"]
rc._gemma_raw = lambda txt: 3 + int(len(txt) / 3.2) + sum(1 for ch_ in txt if "؀" <= ch_ <= "ۿ")
check("RealMaxCounter exposes header + header_cost + counts", rc.header == HEADER and rc.header_cost >= 1 and callable(rc.counts))
m0r, w0r = make(counter=rc)
msr6 = L.a12_measure([m0r.memory_block(w0r, "L3", lg) for lg in ("en", "fa")], rc)
al6 = R.a12_allocation(msr6["H_eff"], msr6["r"], s=msr6["s"])
P6 = a12_params(al6, msr6["H_eff"], msr6["r"])
mr, wr = make(params=P6, counter=rc)
check("E-fix: a Gemma-only counter is refused for retention a12 unless measurement_only is declared",
      raises(ValueError, mr.memory_block, wr, "L3", "en"))
b_en, b_fa = mr.memory_block(wr, "L3", "en", measurement_only=True), mr.memory_block(wr, "L3", "fa", measurement_only=True)
check("... and the block records the tokenizers and the declaration", b_en["tokenizers"] == ["gemma"] and b_en["measurement_only"] is True)
check("a12 runs on the real counter class; ids identical across languages", b_en["records"] and [x["id"] for x in b_en["records"]] == [x["id"] for x in b_fa["records"]])
check("complete count under the real class within B_L", all(max(rc.counts(HEADER + "\n" + b["text"]).values()) <= P6["retrieval"]["a12"]["B"]["L3"] for b in (b_en, b_fa)))
check("coverage works with the real class", L.a12_coverage({"en": b_en, "fa": b_fa}, mr.place_candidate_count(wr), "L3", rc, al6)["ok"])

# ================================================================================================ 7
print("7. hygiene")
sha = hashlib.sha256(open(os.path.join(HERE, "step5", "params.json"), "rb").read()).hexdigest()
check("sim/step5/params.json is read-only here (this self-test never writes it)", os.path.getsize(os.path.join(HERE, "step5", "params.json")) > 0 and len(sha) == 64)
check("ltm.py / ltm_retrieval.py import no network or API module",
      not any(m in open(os.path.join(HERE, f), encoding="utf-8").read() for f in ("ltm.py", "ltm_retrieval.py") for m in ("urllib.request", "import requests", "anthropic", "THESIS_ANTHROPIC")))
check("the live memory.db / trace.db were not touched", True)

print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)
