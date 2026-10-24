"""E4 - the context-fit gate of the ft-v2 prompts (FREEZE-READINESS E4; prereg A24 item 1 and item 7(i); Astra C4-open Q-c).

A24 item 1: "Gemma's num_ctx stays 8192: the full prompt plus the package plus two repair turns plus the 768-token output must fit, and this
is measured on the real prompts (a context-fit gate), not estimated." A24 item 7(i): "measured full-context fit for every Gemma presentation,
including two repair turns."

ONE check = one captured presentation x one prompt source (none / S* package / a curve snapshot / FS) x one model that sees that source. The
messages are the exact bytes of the ONE renderer (plan.render_messages). The worst-case conversation is

    system + user  +  REPAIRS x [ assistant reply at the model's maximum output  +  REPAIR_V2 frame with a problem text of P chars ]
                   +  the final output reserve (the model's maximum output),

and it fits when it is <= limit - margin(limit), margin = max(256, 5 % of the limit) (Gemma: 7,782 of 8,192).

Two modes (Astra: "a forecast plus a measured gate"):
  MEASURED  (--real; the Gemma server counts the WHOLE serialized worst-case conversation in one request, prompt_eval_count with num_ctx
            raised so nothing is truncated, num_predict 1; no planning): verdict PASS_MEASURED / FAIL_MEASURED.
  FORECAST  (offline): per-text counts from the real-token caches where the exact text was counted (forecast-grade only: the caches are keyed
            by text, not by model / template version) or a conservative character estimate; recorded block counts (header + block, max
            over tokenizers, enforced <= 2,000 at capture) are known. Verdicts PASS_FORECAST / UNRESOLVED_EST (unresolved because estimates are
            inside; not "fails only because of estimates") / FAIL_KNOWN (the known part plus the mandatory output reserves already exceed).
FREEZE GRADE needs every Gemma check PASS_MEASURED **and** an enforced cap on the repair feedback (Astra c5 / c6). Since FREEZE-GATES
step 2 (2026-10-13) the runner enforces feedback contract fb-1 (planners.runner.cap_feedback: ASCII fold, <= 600 chars incl. the marker);
`feedback_cap()` reads planners.runner.PROBLEM_TEXT_CAP_CHARS and each repair's problem text is then taken at its TOKEN bound
(planners.runner.problem_token_bound). Without a declared cap the gate is never freeze-grade and the forecast uses P_BOUND_CHARS
(plan.REPAIR_PROBLEM_CHARS) as a stated, conditional character bound. API models (Opus, gpt-6-sol) are forecasts against labelled ASSUMED context floors, never a verified fit.
`journal_fit` is supplementary monitoring for 7(i) (Astra: not its operationalisation): per-attempt prompt_eval_count of the plan journal
against the allowed context and against the preflight's expectation; missing usage is UNKNOWN, not zero.

    python sim/final_test/matrix/context_fit.py --forecast                          # offline: the worst presentation the rules allow
    python sim/final_test/matrix/context_fit.py --namespace rehearsal-v3 [--write]   # offline forecast on an ft-v2 namespace's captures
    python sim/final_test/matrix/context_fit.py --namespace rehearsal-v3 --real --ollama http://127.0.0.1:11434 --write   # measured (Gemma)
    python sim/final_test/matrix/context_fit.py --journal rehearsal-v3               # supplementary: the plan journal's real counts
Python 3.10 syntax; the simulator is never imported.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402  (also puts sim/ on sys.path)

SCHEMA = "final-test.context_fit/2"
REPAIRS = 2                                           # A24 item 2 (= treatment.FT2["repairs"]; checked by the selftest)
#: plan.REPAIR_PROBLEM_CHARS: the forecast's CONDITIONAL problem-text bound while the runner enforces no cap (checked by the selftest)
P_BOUND_CHARS = 600
#: forecast only: chat-template tokens per message (role markers, turn delimiters). The measured mode counts the real template.
TEMPLATE_PER_MESSAGE = 8
#: model -> context limit, max output, the tokenizer used for forecasting, provenance. Only Gemma's are measured settings (read from the provider).
MODELS = {
    "gemma": {"limit": None, "output": None, "tokenizer": "gemma", "assumed": False,
              "source": "planners/ollama_provider.OPTIONS num_ctx / num_predict (A24 item 1: num_ctx stays 8192)"},
    "opus": {"limit": 200000, "output": 8000, "tokenizer": "anthropic", "assumed": True,
             "source": "ASSUMED conservative floor 200,000 tokens; output = anthropic_provider max_tokens 8000 (forecast only, never a verified fit)"},
    "codex": {"limit": 128000, "output": 8000, "tokenizer": "max", "assumed": True,
              "source": "ASSUMED conservative floor 128,000 for gpt-6-sol via the Codex CLI; counted as the max of both tokenizers (forecast only)"},
}
PLANNER_MODEL = {"G": "gemma", "O": "opus", "X": "codex"}
#: the conservative character estimator (tokens per character, + a constant per text); calibrated against the real caches (`calibration`)
EST = {"gemma": (0.50, 8), "anthropic": (0.80, 8)}
VERDICTS = ("FAIL_MEASURED", "FAIL_KNOWN", "UNRESOLVED_EST", "PASS_FORECAST", "PASS_MEASURED")      # worst first
GEMMA_MODEL = "gemma4:e2b-it-q4_K_M"


class ContextFitError(RuntimeError):
    pass


def _gemma_settings():
    if MODELS["gemma"]["limit"] is None:
        from planners.ollama_provider import OPTIONS
        MODELS["gemma"]["limit"], MODELS["gemma"]["output"] = int(OPTIONS["num_ctx"]), int(OPTIONS["num_predict"])
    return MODELS["gemma"]


def model(m: str) -> dict:
    if m == "gemma":
        _gemma_settings()
    return MODELS[m]


def margin(limit: int) -> int:
    return max(256, int(math.ceil(0.05 * int(limit))))


def allowed(m: str) -> int:
    lim = int(model(m)["limit"])
    return lim - margin(lim)


def feedback_cap():
    """The ENFORCED cap on the repair problem text, if the runner declares one (planners.runner.PROBLEM_TEXT_CAP_CHARS); None = uncapped."""
    try:
        from planners import runner as R
    except Exception:                                 # noqa: BLE001
        return None
    v = getattr(R, "PROBLEM_TEXT_CAP_CHARS", None)
    return int(v) if isinstance(v, int) and v > 0 else None


def _fb_contract():
    if feedback_cap() is None:
        return None
    from planners import runner as R
    return getattr(R, "FEEDBACK_CONTRACT", None)


def est_tokens(text: str, tokenizer: str) -> int:
    """Conservative token estimate of one text (forecast only)."""
    if not text:
        return 0
    r, k = EST[tokenizer]
    return int(math.ceil(r * len(text))) + k


# ================================================================================================ the real-token caches (forecast-grade)
def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cache_files(root: str | None = None) -> list:
    r = C.paths(root)["root"]
    out = []
    for p in (os.path.join(r, "sim", "final_test", "**", "tokcache*.json"), os.path.join(r, "docs", "final-test", "**", "tokcache*.json")):
        out += glob.glob(p, recursive=True)
    return sorted(set(out))


def load_caches(paths) -> dict:
    """{text-sha24: {"gemma": net, "anthropic": net, "chars": n}}, net = raw - raw('.') (RealMaxCounter.counts semantics: a single-user-message
    difference, NOT an additive full-conversation count; forecast-grade only)."""
    out = {}
    k0 = _key(".")
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        b = d.get(k0) or {}
        if "gemma_raw" not in b:
            continue
        for k, v in d.items():
            if k == k0 or not isinstance(v, dict) or "gemma_raw" not in v:
                continue
            e = {"chars": int(v.get("chars") or 0), "gemma": int(v["gemma_raw"]) - int(b["gemma_raw"])}
            if "anth_raw" in v and "anth_raw" in b:
                e["anthropic"] = int(v["anth_raw"]) - int(b["anth_raw"])
            out[k] = e
    return out


def calibration(cache: dict, min_chars: int = 1) -> dict:
    """Is est_tokens >= (net count + 1) for every cached text? The estimator's honesty check on the texts we have real counts for. It does
    NOT bound arbitrary unseen text (validator output, short strings): that is why a forecast is never freeze-grade."""
    worst = {"gemma": None, "anthropic": None}
    bad = {"gemma": 0, "anthropic": 0}
    n = 0
    for e in cache.values():
        if e["chars"] < min_chars:
            continue
        n += 1
        for tk in ("gemma", "anthropic"):
            if tk not in e:
                continue
            r, c = EST[tk]
            est = int(math.ceil(r * e["chars"])) + c
            real = e[tk] + 1
            if worst[tk] is None or est - real < worst[tk]["slack"]:
                worst[tk] = {"slack": est - real, "chars": e["chars"], "real": real, "est": est}
            bad[tk] += int(real > est)
    return {"texts": n, "violations": bad, "tightest": worst, "ok": n > 0 and not any(bad.values())}


class Counter:
    """Forecast counts per text (cache -> 'known', else 'est') and, when `real_messages(messages) -> {"gemma": prompt_eval_count}` is given,
    the MEASURED count of a whole serialized conversation."""

    def __init__(self, cache: dict | None = None, real_messages=None):
        self.cache = cache or {}
        self.real_messages = real_messages

    def text(self, text: str, tokenizer: str) -> tuple:
        if tokenizer == "max":
            a, ka = self.text(text, "gemma")
            b, kb = self.text(text, "anthropic")
            return max(a, b), ("known" if ka == kb == "known" else "est")
        if not text:
            return 0, "known"
        e = self.cache.get(_key(text))
        if e and tokenizer in e:
            return int(e[tokenizer]) + 1, "known"
        return est_tokens(text, tokenizer), "est"


# ================================================================================================ one check
def repair_text(p_chars: int) -> str:
    from planners.contract import REPAIR_V2
    return REPAIR_V2.format(problems="-" * int(p_chars))


def repair_frame() -> str:
    """The REPAIR_V2 frame with an EMPTY problem text (fb-1 forecast: frame counted, the problem text added at its token bound)."""
    from planners.contract import REPAIR_V2
    return REPAIR_V2.format(problems="")


def problem_bound(tokenizer: str):
    """fb-1: the token bound of one capped problem text under `tokenizer` (planners.runner.problem_token_bound); None = uncapped."""
    if feedback_cap() is None:
        return None
    from planners import runner as R
    return int(R.problem_token_bound(tokenizer))


def reply_filler(counter: Counter, n_tokens: int, role: str = "assistant") -> str:
    """MEASURED mode: a text whose real Gemma count (as a message of `role`) is >= n_tokens (a bounded replayed reply, or with
    role="user" the fb-1 problem-text stand-in at its proven token bound)."""
    k = max(1, int(n_tokens))
    for _ in range(6):
        t = " ".join(["ok"] * k)
        got = int(counter.real_messages([{"role": role, "content": t}])["gemma"])
        base = int(counter.real_messages([{"role": role, "content": "ok"}])["gemma"])
        if got - base + 1 >= n_tokens:
            return t
        k = int(math.ceil(k * (n_tokens / max(1, got - base + 1)) * 1.05)) + 1
    raise ContextFitError("could not build a filler of %d tokens" % n_tokens)


def fit(msgs: list, mdl: str, counter: Counter, *, block_tokens: int | None = None, block_text: str | None = None,
        repairs: int = REPAIRS, p_chars: int | None = None, filler: str | None = None, pfiller: str | None = None) -> dict:
    """The worst-case conversation of one request under one model. Forecast: `block_tokens` (+ the exact `block_text` inside the user
    message) is a block whose count is already known (recorded at capture); the request is counted without it and the known count added.
    fb-1 (the runner declares its cap): each repair's problem text is taken at its TOKEN bound (planners.runner.problem_token_bound:
    Gemma 600, proven; the API routes 2 x 600), not at a character filler; measured mode replays a problem stand-in whose real Gemma
    count is >= that bound."""
    m = model(mdl)
    cap = feedback_cap()
    p = int(cap if cap is not None else (P_BOUND_CHARS if p_chars is None else p_chars))
    out_t, lim = int(m["output"]), allowed(mdl)
    tk = m["tokenizer"]
    pb = problem_bound(tk)
    rep_txt = repair_text(p)
    res = {"model": mdl, "repairs": repairs, "output": out_t, "limit": int(m["limit"]), "margin": margin(m["limit"]), "allowed": lim,
           "p_chars": p, "feedback_capped": cap is not None, "problem_token_bound": pb, "assumed_context": bool(m.get("assumed"))}
    if counter.real_messages is not None and mdl == "gemma":
        fil = filler if filler is not None else reply_filler(counter, out_t)
        if pb is not None:
            from planners.contract import REPAIR_V2
            rep_txt = REPAIR_V2.format(problems=pfiller if pfiller is not None else reply_filler(counter, pb, role="user"))
        conv = list(msgs)
        for _ in range(repairs):
            conv += [{"role": "assistant", "content": fil}, {"role": "user", "content": rep_txt}]
        prompt = int(counter.real_messages(conv)["gemma"])
        total = prompt + out_t
        ok = total <= lim
        res.update(mode="measured", prompt=prompt, total=total, headroom=lim - total, fits=ok, verdict="PASS_MEASURED" if ok else "FAIL_MEASURED")
        return res
    use = msgs
    known = est = 0
    if block_tokens is not None and block_text:
        u = msgs[1]["content"]
        if u.count(block_text) != 1:
            raise ContextFitError("the block text is not inside the user message exactly once")
        use = [msgs[0], dict(msgs[1], content=u.replace(block_text, "", 1))]
        known += int(block_tokens)
    for mm in use:
        n, k = counter.text(mm["content"], tk)
        if k == "known":
            known += n
        else:
            est += n
    if pb is not None:
        rn, rk = counter.text(repair_frame(), tk)
        known += repairs * pb                                      # the capped problem text at its token bound: known by rule
    else:
        rn, rk = counter.text(rep_txt, tk)
    n_msgs = len(use) + 2 * repairs
    tmpl = n_msgs * TEMPLATE_PER_MESSAGE
    outs = (repairs + 1) * out_t                                   # the replayed replies at the maximum + the final output: known by rule
    rep_part = repairs * rn
    if rk == "known":
        known += rep_part
    else:
        est += rep_part
    total = known + est + tmpl + outs
    if known + outs > lim:
        v = "FAIL_KNOWN"
    elif total <= lim:
        v = "PASS_FORECAST"
    else:
        v = "UNRESOLVED_EST"
    res.update(mode="forecast", known=known, estimated=est, template=tmpl, outputs=outs, total=total, headroom=lim - total,
               fits=total <= lim, verdict=v)
    return res


def worst_verdict(vs) -> str | None:
    vs = list(vs)
    for v in VERDICTS:
        if v in vs:
            return v
    return None


# ================================================================================================ the gate over captures
def models_for_source(src: str, ns: str) -> list:
    """Which models see this source: the namespace's main arms whose prompt is `src`; a curve snapshot -> Gemma and Opus (G+M@ / O+M@)."""
    out = set()
    for arm in C.main_arms_for(ns):
        info = C.arm_info(arm, ns)
        if info["prompt"] == src:
            out.add(PLANNER_MODEL.get(info["planner"][0]))
    if not out and src not in ("none", "S*", "FS"):
        out = {"gemma", "opus"}
    return sorted(m for m in out if m)


def _block(rec: dict, src: str):
    """(known block tokens incl. header, the exact header + text inside the user message) or (None, None)."""
    from planners.contract import EXPERIENCE_HEADER, PACKAGE_HEADER
    blk = (rec.get("blocks") or {}).get(src) or {}
    text = blk.get("text") or ""
    if not text or len(blk.get("counts") or {}) < 2:
        return None, None
    hdr = EXPERIENCE_HEADER if src == "FS" else PACKAGE_HEADER
    return int(blk.get("real_tokens") or 0) + TEMPLATE_PER_MESSAGE, hdr + "\n" + text


def gate_rows(rows: list, contract, counter: Counter, ns: str, *, repairs: int = REPAIRS, p_chars: int | None = None) -> dict:
    """E4 over an ft-v2 namespace's run-1 capture rows: every presentation x source x model (curve snapshots included)."""
    import plan as PLN
    checks, filler, pfiller = [], None, None
    for r in rows:
        if r.get("status") not in (None, "ok") or int(r.get("capture_run", 1) or 1) != 1:
            continue
        if r.get("treatment") != "ft-v2":
            raise ContextFitError("%s: not an ft-v2 capture row" % r.get("pres"))
        for src in sorted(r.get("blocks") or {}):
            msgs = PLN.render_messages(contract, r, src)
            bt, btxt = _block(r, src)
            for mdl in models_for_source(src, ns):
                if counter.real_messages is not None and mdl == "gemma" and filler is None:
                    filler = reply_filler(counter, int(model("gemma")["output"]))
                    pb = problem_bound("gemma")
                    pfiller = reply_filler(counter, pb, role="user") if pb is not None else None
                f = fit(msgs, mdl, counter, block_tokens=bt, block_text=btxt, repairs=repairs, p_chars=p_chars, filler=filler,
                        pfiller=pfiller)
                checks.append(dict(f, pres=r.get("pres"), source=src, lang=r.get("lang"), level=r.get("level"),
                                   request_sha=PLN.prompt_sha256(msgs)))      # ACC-RUNNER c3: binds the check to the measured request bytes
    return summarise(checks)


def summarise(checks: list) -> dict:
    by_model = {}
    for c in checks:
        s = by_model.setdefault(c["model"], {"n": 0, "verdicts": {v: 0 for v in VERDICTS}, "worst": None, "assumed_context": c["assumed_context"]})
        s["n"] += 1
        s["verdicts"][c["verdict"]] += 1
        if s["worst"] is None or c["total"] > s["worst"]["total"]:
            s["worst"] = {k: c.get(k) for k in ("pres", "source", "lang", "total", "allowed", "headroom", "mode", "verdict")}
    gem = [c["verdict"] for c in checks if c["model"] == "gemma"]
    capped = feedback_cap() is not None
    why = []
    if not gem:
        why.append("no Gemma check")
    elif any(v != "PASS_MEASURED" for v in gem):
        why.append("not every Gemma check is PASS_MEASURED (worst: %s)" % worst_verdict(gem))
    if not capped:
        why.append("the repair feedback is UNCAPPED (planners.runner.PROBLEM_TEXT_CAP_CHARS absent): no measured fit holds for every reply")
    return {"schema": SCHEMA, "n_checks": len(checks), "verdict": worst_verdict(c["verdict"] for c in checks),
            "gemma_verdict": worst_verdict(gem), "freeze_grade": not why, "not_freeze_grade_because": why, "by_model": by_model,
            "rule": {"repairs": REPAIRS, "p_bound_chars": P_BOUND_CHARS, "feedback_cap": feedback_cap(), "feedback_contract": _fb_contract(),
                     "problem_token_bound": {t: problem_bound(t) for t in ("gemma", "anthropic", "max")},
                     "template_per_message_forecast": TEMPLATE_PER_MESSAGE,
                     "margin": "max(256, 5 % of the limit)", "models": {k: model(k) for k in MODELS},
                     "estimator": {k: {"per_char": v_[0], "plus": v_[1]} for k, v_ in EST.items()}},
            "checks": checks}


# ================================================================================================ forecast (no ft-v2 capture exists yet)
def synthetic_presentation(types: list, *, lang: str = "fa") -> dict:
    """A deliberately long presentation: 40 summary lines over the longest type names and a long instruction. Diagnostic only; never an item."""
    lt = sorted(types, key=lambda t: (-len(t), t))
    summary = ["%s [open, filled, toggled on] seen at (%d.%02d, -%d.%02d), step %d; inside: %s" % (lt[i % len(lt)], i % 5, i, i % 4, 10 + i, i,
                                                                                                    lt[(i + 7) % len(lt)]) for i in range(40)]
    instr = ("لطفاً سیب را از یخچال بردار، آن را در سینک بشوی، بعد آن را روی پیشخوان کنار قهوه‌ساز بگذار و در یخچال را ببند"
             if lang == "fa" else
             "Please take the apple out of the fridge, wash it in the sink, then put it on the counter next to the coffee machine and close the fridge")
    return {"instruction": instr, "prompt_summary": summary, "holding": None, "lang": lang}


def forecast(types: list, counter: Counter, *, block_tokens: int = 2000, p_chars: int | None = None) -> dict:
    """The worst presentation the frozen rules allow: the ft-v2 system text with the shown type list, 40 long summary lines, a long
    instruction, and a block at the 2,000-token maximum (a hard bound enforced at capture under every tokenizer)."""
    import treatment as T
    import honest_skills as hs
    from planners.contract import PlannerContract, EXPERIENCE_HEADER
    contract = PlannerContract(T.planner_types(types), tools=hs.TOOLS, version=T.FT2_VERSION)
    out = []
    for lang in ("en", "fa"):
        rec = synthetic_presentation(types, lang=lang)
        none = contract.messages(rec["instruction"], rec["prompt_summary"], None)
        withb = contract.messages(rec["instruction"], rec["prompt_summary"], None, fewshot_block="x")
        for mdl in ("gemma", "opus", "codex"):
            out.append(dict(fit(none, mdl, counter, p_chars=p_chars), source="none", lang=lang, pres="synthetic"))
            out.append(dict(fit(withb, mdl, counter, block_tokens=block_tokens + TEMPLATE_PER_MESSAGE, block_text=EXPERIENCE_HEADER + "\nx",
                                p_chars=p_chars), source="block@2000", lang=lang, pres="synthetic"))
    res = summarise(out)
    sysn, sysk = counter.text(contract.system_text(), "gemma")
    res["system_text"] = {"chars": len(contract.system_text()), "gemma_tokens": sysn, "kind": sysk}
    return res


def adversarial_problem_chars(types: list, steps=(12, 24, 48, 80)) -> dict:
    """What validator_v2 + runner.problems_text produce for an invalid reply of n steps (unknown tool, unknown types, bad arg on every step).
    The runner passes ALL of it to the repair turn."""
    from planners import validator_v2 as V
    from planners.contract import problems_text
    out = {}
    for n in steps:
        plan = [{"action": "pour_%d" % i, "object": "NotAType%d" % i, "to": "NotAType%d" % i, "arg": "lava"} for i in range(n)]
        v = V.validate({"plan": plan}, types, None)
        out[str(n)] = {"errors": len(v["errors"]), "chars": len(problems_text(v["errors"]))}
    return out


# ================================================================================================ supplementary: the plan journal (7(i) monitoring)
def journal_fit(rows: list, expected: dict | None = None, m: str = "gemma") -> dict:
    """Every recorded attempt of a Gemma cell: tokens_in (the server's prompt_eval_count of THAT call) + the output budget <= the allowed
    context; with `expected` ({pres|source: preflight base tokens}) attempt k is also reconciled against base + (k-1) x a bounded repair
    turn. Missing usage is UNKNOWN (never 0). Monitoring only: first calls that succeed never exercise the second repair (Astra)."""
    lim, out_b = allowed(m), int(model(m)["output"])
    n, over, unknown, above_exp, worst = 0, [], [], [], None
    for r in rows:
        if not str(r.get("planner") or "").startswith("G"):
            continue
        for a in r.get("attempts") or []:
            ti = a.get("tokens_in")
            if not ti:
                unknown.append({"cell": r.get("key") or r.get("cell"), "k": a.get("k")})
                continue
            ti = int(ti)
            n += 1
            tot = ti + out_b
            row = {"cell": r.get("key") or r.get("cell"), "k": a.get("k"), "tokens_in": ti, "total": tot}
            if worst is None or tot > worst["total"]:
                worst = row
            if tot > lim:
                over.append(row)
            if expected:
                e = expected.get("%s|%s" % (r.get("pres"), r.get("source")))
                if e is not None and ti > e["base"] + (int(a.get("k") or 0)) * e["per_turn"]:
                    above_exp.append(dict(row, expected=e))
    return {"model": m, "calls": n, "allowed": lim, "over": over, "unknown_usage": unknown, "above_preflight": above_exp, "worst": worst,
            "ok": n > 0 and not over and not unknown, "role": "supplementary monitoring (not the 7(i) operationalisation)"}


# ================================================================================================ measured counter (--real; the Gemma server)
def gemma_messages_counter(endpoint: str, mdl: str = GEMMA_MODEL, num_ctx: int = 32768):
    """prompt_eval_count of a whole conversation; num_ctx raised far above 8,192 so the count itself is never truncated; num_predict 1."""
    def count(msgs):
        body = {"model": mdl, "messages": msgs, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 7, "num_ctx": int(num_ctx), "num_predict": 1, "top_k": 1}}
        req = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
        return {"gemma": int(json.loads(urllib.request.urlopen(req, timeout=300).read())["prompt_eval_count"])}
    return count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4 context-fit gate (A24 items 1 and 7(i))")
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--namespace")
    ap.add_argument("--journal")
    ap.add_argument("--real", action="store_true", help="MEASURED: the Gemma server counts each whole worst-case conversation (no planning)")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    import plan as PLN
    cache = load_caches(cache_files(a.root))
    counter = Counter(cache, real_messages=gemma_messages_counter(a.ollama) if a.real else None)
    types, _src = PLN.load_types(None, a.root)
    if a.journal:
        C.ns_check(a.journal)
        res = journal_fit(C.Journal(C.journal_path(a.journal, "plan", a.root)).rows())
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 4
    if a.forecast:
        res = forecast(types, counter)
        res["calibration"] = calibration(cache)
        res["adversarial_problem_chars"] = adversarial_problem_chars(types)
        print(json.dumps({k: v for k, v in res.items() if k != "checks"}, indent=1, ensure_ascii=False))
        return 0
    if not a.namespace:
        ap.error("--forecast, --namespace or --journal")
    import treatment as T
    if not T.is_ft2(a.namespace):
        raise SystemExit("E4 is the ft-v2 gate; %s is an ft-v1 namespace" % a.namespace)
    contract = PLN.make_contract(types, a.namespace)
    cap_path = C.journal_path(a.namespace, "capture", a.root)
    res = gate_rows(C.Journal(cap_path).rows(), contract, counter, a.namespace)
    res.update(calibration=calibration(cache), namespace=a.namespace, capture_sha=C.sha256_file(cap_path),   # ACC-RUNNER c3: binding
               counting="MEASURED (Gemma server, whole conversation)" if a.real else "FORECAST (caches + conservative estimate)")
    if a.write:
        p = os.path.join(C.ns_docs(a.namespace, a.root), "context_fit.json")
        text = json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True)
        if C.key_leak(text):
            raise SystemExit("key-like text in the output: not written")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        print("wrote", p)
    print(json.dumps({k: v for k, v in res.items() if k != "checks"}, indent=1, ensure_ascii=False))
    return 0 if res["freeze_grade"] else (4 if res["verdict"] in ("FAIL_MEASURED", "FAIL_KNOWN") else 5)


if __name__ == "__main__":
    sys.exit(main())
