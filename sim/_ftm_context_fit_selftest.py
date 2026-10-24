"""C4-open gate (c): E4 context fit (sim/final_test/matrix/context_fit.py; prereg A24 items 1 and 7(i); Astra C4-open Q-c).
Offline: Windows python, repo root; no simulator, no LLM, no network (the measured mode runs on a FAKE whole-conversation counter).

    python sim/_ftm_context_fit_selftest.py
"""
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                     # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                                    # noqa: E402
import context_fit as F                                               # noqa: E402
import treatment as T                                                 # noqa: E402
import plan as PLN                                                    # noqa: E402
from planners import runner as R                                      # noqa: E402
from planners.contract import PlannerContract, EXPERIENCE_HEADER, PACKAGE_HEADER   # noqa: E402
from planners.ollama_provider import OPTIONS                          # noqa: E402
import honest_skills as hs                                            # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)[:400]))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


TYPES49 = json.load(open(os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]
c2 = PlannerContract(T.planner_types(TYPES49), tools=hs.TOOLS, version="ft-v2")

print("== 1 constants")
check("1.1 two repairs (= treatment.FT2), forecast bound = plan.REPAIR_PROBLEM_CHARS", F.REPAIRS == T.FT2["repairs"] == 2 and F.P_BOUND_CHARS == PLN.REPAIR_PROBLEM_CHARS)
check("1.2 Gemma limit / output READ from the provider (8192 / 768); margin max(256, 5 %) -> 410, allowed 7,782",
      F.model("gemma")["limit"] == OPTIONS["num_ctx"] == 8192 and F.model("gemma")["output"] == OPTIONS["num_predict"] == 768
      and F.margin(8192) == 410 and F.allowed("gemma") == 7782 and F.margin(1000) == 256)
check("1.3 API context sizes are labelled ASSUMED floors", F.MODELS["opus"]["assumed"] and F.MODELS["codex"]["assumed"] and not F.MODELS["gemma"]["assumed"])
check("1.4 fb-1: the runner declares its feedback cap (600 chars incl. marker, contract fb-1); token bounds gemma 600 / anthropic 1200 / max 1200",
      F.feedback_cap() == R.PROBLEM_TEXT_CAP_CHARS == 600 and R.FEEDBACK_CONTRACT == "fb-1"
      and (F.problem_bound("gemma"), F.problem_bound("anthropic"), F.problem_bound("max")) == (600, 1200, 1200))
#: sections 3-5 test the arithmetic of an UNCAPPED runner (the pre-fb-1 path stays supported); section 8 tests fb-1
_CAP = R.PROBLEM_TEXT_CAP_CHARS
R.PROBLEM_TEXT_CAP_CHARS = None

print("== 2 estimator calibration on the real-token caches")
cache = F.load_caches(F.cache_files())
cal = F.calibration(cache)
check("2.1 the caches hold real counts of many texts (both tokenizers)", cal["texts"] > 500, cal["texts"])
check("2.2 the character estimate is >= the real count of EVERY cached text (forecast honesty)", cal["ok"], cal)

print("== 3 forecast arithmetic")
fake_cache = {}
sysm = c2.messages("pick up the apple", ["Apple [?] on/in CounterTop at (1.00, 2.00), step 3"], None)
for m in sysm:
    fake_cache[F._key(m["content"])] = {"gemma": 99, "anthropic": 199, "chars": len(m["content"])}
kc = F.Counter(fake_cache)
f1 = F.fit(sysm, "gemma", kc)
rep_n = F.est_tokens(F.repair_text(F.P_BOUND_CHARS), "gemma")
check("3.1 known texts from the cache; repair text estimated; outputs = 3 x 768; template 6 messages x 8",
      f1["mode"] == "forecast" and f1["known"] == 2 * 100 and f1["estimated"] == 2 * rep_n and f1["outputs"] == 3 * 768
      and f1["template"] == 6 * 8 and f1["total"] == 200 + 2 * rep_n + 48 + 2304, f1)
check("3.2 it fits -> PASS_FORECAST (never PASS_MEASURED offline)", f1["verdict"] == "PASS_FORECAST")
big = c2.messages("x " * 9000, [], None)
f2 = F.fit(big, "gemma", F.Counter({}))
check("3.3 estimates over the limit -> UNRESOLVED_EST (unresolved, not 'fails only because of estimates')", f2["verdict"] == "UNRESOLVED_EST", f2["verdict"])
fake_big = {F._key(m["content"]): {"gemma": 6000, "anthropic": 6000, "chars": 1} for m in big}
f3 = F.fit(big, "gemma", F.Counter(fake_big))
check("3.4 the KNOWN part + the output reserves already exceed -> FAIL_KNOWN", f3["verdict"] == "FAIL_KNOWN", f3)
withb = c2.messages("pick up the apple", [], None, memory_block="AAAA")
f4 = F.fit(withb, "gemma", F.Counter({}), block_tokens=1500, block_text=PACKAGE_HEADER + "\nAAAA")
f5 = F.fit(c2.messages("pick up the apple", [], None), "gemma", F.Counter({}))
check("3.5 a recorded block count is used as KNOWN (the block text is not re-estimated)", f4["known"] == 1500 and abs(f4["estimated"] - f5["estimated"]) <= 4, (f4, f5))
check("3.6 a block text not inside the user message exactly once is an error",
      raises(lambda: F.fit(withb, "gemma", F.Counter({}), block_tokens=10, block_text="ZZZ"), F.ContextFitError))
fo = F.fit(sysm, "opus", F.Counter({}))
check("3.7 Opus: 8,000-token outputs, the 200k ASSUMED floor, forecast only", fo["output"] == 8000 and fo["assumed_context"] and fo["verdict"] == "PASS_FORECAST")

print("== 4 measured mode (fake whole-conversation counter)")
seen = []


def fake_real(msgs):
    seen.append(len(msgs))
    return {"gemma": sum(len(m["content"]) // 4 + 5 for m in msgs)}


mc = F.Counter({}, real_messages=fake_real)
fil = F.reply_filler(mc, 768)
check("4.1 the reply filler counts >= 768 tokens as a message", fake_real([{"role": "assistant", "content": fil}])["gemma"]
      - fake_real([{"role": "assistant", "content": "ok"}])["gemma"] + 1 >= 768)
seen.clear()
m1 = F.fit(sysm, "gemma", mc, filler=fil)
check("4.2 ONE count of the whole serialized conversation: system, user, 2 x (assistant reply, repair) = 6 messages", seen == [6], seen)
check("4.3 total = measured prompt + one output reserve; PASS_MEASURED when inside 7,782", m1["mode"] == "measured" and m1["total"] == m1["prompt"] + 768
      and m1["verdict"] == "PASS_MEASURED", m1)
m2 = F.fit(big, "gemma", mc, filler=fil)
check("4.4 over the allowed context -> FAIL_MEASURED", m2["verdict"] == "FAIL_MEASURED")
mo = F.fit(sysm, "opus", mc, filler=fil)
check("4.5 the measured counter is Gemma's: Opus stays a forecast", mo["mode"] == "forecast")

print("== 5 the gate over ft-v2 capture rows")


def row(pres, blocks, treatment="ft-v2"):
    return {"pres": pres, "status": "ok", "capture_run": 1, "treatment": treatment, "lang": "en", "level": "L1", "instruction": "pick up the apple",
            "prompt_summary": ["Apple [?] on/in CounterTop at (1.00, 2.00), step 3"], "summary": [], "holding": None, "blocks": blocks}


B = {"none": {"text": ""}, "S*": {"text": "pkg line", "real_tokens": 40, "counts": {"gemma": 30, "anthropic": 40}},
     "FS": {"text": "1. en | x | pickup Apple | ok | success", "real_tokens": 50, "counts": {"gemma": 40, "anthropic": 50}},
     "o0k3": {"text": "", "real_tokens": 0, "counts": {}}}
g = F.gate_rows([row("p1:en", B)], c2, F.Counter({}), "rehearsal-v3")
srcs = {}
for c in g["checks"]:
    srcs.setdefault(c["source"], set()).add(c["model"])
check("5.1 none / S* -> Gemma, Opus, gpt-6-sol; FS -> Gemma, Opus; a curve snapshot -> Gemma, Opus",
      srcs == {"none": {"gemma", "opus", "codex"}, "S*": {"gemma", "opus", "codex"}, "FS": {"gemma", "opus"}, "o0k3": {"gemma", "opus"}}, srcs)
check("5.2 the rendered request is plan.render_messages' (the S* known block = its recorded count + the join allowance)",
      next(c for c in g["checks"] if c["source"] == "S*" and c["model"] == "gemma")["known"] == 40 + F.TEMPLATE_PER_MESSAGE)
check("5.3 an offline gate is never freeze-grade; both reasons are stated", not g["freeze_grade"] and len(g["not_freeze_grade_because"]) == 2, g["not_freeze_grade_because"])
check("5.4 an ft-v1 capture row is refused", raises(lambda: F.gate_rows([row("p:en", B, treatment=None)], c2, F.Counter({}), "rehearsal-v3"), F.ContextFitError))
gm = F.gate_rows([row("p1:en", B)], c2, mc, "rehearsal-v3")
check("5.5 measured Gemma checks + an UNCAPPED runner -> still not freeze-grade (feedback reason only)",
      gm["gemma_verdict"] == "PASS_MEASURED" and not gm["freeze_grade"] and len(gm["not_freeze_grade_because"]) == 1)
R.PROBLEM_TEXT_CAP_CHARS = 400
try:
    gc = F.gate_rows([row("p1:en", B)], c2, mc, "rehearsal-v3")
    check("5.6 once the runner DECLARES a cap, it becomes the bound (400 chars) and measured passes are freeze-grade",
          gc["freeze_grade"] and all(c["p_chars"] == 400 and c["feedback_capped"] for c in gc["checks"]))
finally:
    R.PROBLEM_TEXT_CAP_CHARS = None
check("5.7 summarise: the worst verdict order FAIL_MEASURED > FAIL_KNOWN > UNRESOLVED_EST > PASS_FORECAST > PASS_MEASURED",
      F.worst_verdict(["PASS_MEASURED", "PASS_FORECAST"]) == "PASS_FORECAST" and F.worst_verdict(["UNRESOLVED_EST", "FAIL_KNOWN"]) == "FAIL_KNOWN")

R.PROBLEM_TEXT_CAP_CHARS = _CAP

print("== 6 supplementary journal monitoring")
jr = [{"planner": "G", "key": "c1", "pres": "p1:en", "source": "none", "attempts": [{"k": 0, "tokens_in": 3000}, {"k": 1, "tokens_in": 7100}]},
      {"planner": "G", "key": "c2", "pres": "p2:en", "source": "none", "attempts": [{"k": 0, "tokens_in": None}]},
      {"planner": "O", "key": "c3", "attempts": [{"k": 0, "tokens_in": 99999}]}]
j = F.journal_fit(jr, expected={"p1:en|none": {"base": 3000, "per_turn": 900}})
check("6.1 a Gemma call within 768 of num_ctx is flagged; Opus rows ignored", [o["cell"] for o in j["over"]] == ["c1"] and j["calls"] == 2)
check("6.2 missing usage is UNKNOWN (not zero) and fails the check", j["unknown_usage"] == [{"cell": "c2", "k": 0}] and not j["ok"])
check("6.3 attempts above the preflight expectation are listed (reconciliation)", [a["cell"] for a in j["above_preflight"]] == ["c1"])
check("6.4 the journal check declares itself supplementary", "supplementary" in j["role"])

print("== 7 the uncapped repair feedback and the offline forecast")
adv = F.adversarial_problem_chars(TYPES49, steps=(12, 48))
check("7.1 the validator's raw problem text grows with the reply's steps and exceeds 600 chars (what fb-1 must cut)",
      adv["48"]["chars"] > adv["12"]["chars"] > F.P_BOUND_CHARS, adv)
fc = F.forecast(TYPES49, F.Counter(cache))
check("7.2 the offline forecast covers EN and FA x none / a 2,000-token block x 3 models; never freeze-grade",
      fc["n_checks"] == 12 and not fc["freeze_grade"] and fc["gemma_verdict"] in ("PASS_FORECAST", "UNRESOLVED_EST", "FAIL_KNOWN"))
print("   INFO forecast: Gemma worst %s total %d of %d allowed (system text %d chars, %s %d tokens)" % (
    fc["by_model"]["gemma"]["worst"]["verdict"], fc["by_model"]["gemma"]["worst"]["total"], F.allowed("gemma"),
    fc["system_text"]["chars"], fc["system_text"]["kind"], fc["system_text"]["gemma_tokens"]))

print("== 8 fb-1 (the runner's cap declared): problem texts at their TOKEN bound")
f8 = F.fit(sysm, "gemma", kc)
fr_n = F.est_tokens(F.repair_frame(), "gemma")
check("8.1 forecast: known = cached texts + 2 x 600 (the proven Gemma bound); the frame alone is estimated",
      f8["feedback_capped"] and f8["problem_token_bound"] == 600 and f8["known"] == 200 + 2 * 600 and f8["estimated"] == 2 * fr_n, f8)
check("8.2 Opus counts 2 x 1,200 (anthropic factor 2.0); gpt-6-sol 2 x 1,200 (max)",
      F.fit(sysm, "opus", F.Counter({}))["problem_token_bound"] == 1200 and F.fit(sysm, "codex", F.Counter({}))["problem_token_bound"] == 1200)
seen.clear()
m8 = F.fit(sysm, "gemma", mc, filler=fil)
pf = F.reply_filler(mc, 600, role="user")
check("8.3 measured: the problem stand-in counts >= 600 real tokens as a message",
      fake_real([{"role": "user", "content": pf}])["gemma"] - fake_real([{"role": "user", "content": "ok"}])["gemma"] + 1 >= 600)
check("8.4 measured: one 6-message conversation count, and it is >= the uncapped dash measurement", 6 in seen and m8["mode"] == "measured"
      and m8["prompt"] >= m1["prompt"], (seen, m8.get("prompt"), m1.get("prompt")))
g8 = F.gate_rows([row("p1:en", B)], c2, mc, "rehearsal-v3")
check("8.5 measured Gemma checks under fb-1 are freeze-grade (no reason left); the rule names fb-1 and the bounds",
      g8["freeze_grade"] and g8["not_freeze_grade_because"] == [] and g8["rule"]["feedback_contract"] == "fb-1"
      and g8["rule"]["problem_token_bound"]["gemma"] == 600, g8["not_freeze_grade_because"])
g8o = F.gate_rows([row("p1:en", B)], c2, F.Counter({}), "rehearsal-v3")
check("8.6 offline under fb-1: never freeze-grade, and the ONLY reason is 'not measured'", not g8o["freeze_grade"]
      and len(g8o["not_freeze_grade_because"]) == 1 and "PASS_MEASURED" in g8o["not_freeze_grade_because"][0], g8o["not_freeze_grade_because"])


print("\n_ftm_context_fit_selftest: %d/%d passed" % (N["pass"], N["pass"] + N["fail"]))
sys.exit(0 if N["fail"] == 0 else 1)
