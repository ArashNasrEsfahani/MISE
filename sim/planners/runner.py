"""PlannerRunner: provider call + whole-plan validation + a bounded number of repair re-prompts (preregistration s4).

Mirrors ``step4_gemma.GemmaPlanner.plan`` on top of any ``PlannerProvider``:
  * the reply is parsed (``contract.parse_reply``), shape-checked against the plan schema, then validated as a whole
    by the validator (injectable ``validate``; the default follows the contract version: ft-v1 -> ``step4_gemma.validate``,
    ft-v2 -> ``planners.validator_v2.validate``);
  * an invalid reply gets a repair: the failed reply + the validator's reasons (tools / types only, no scene fact) go
    back as two more turns, as another provider call with its own cache key (stage="repair"). ft-v1: at most ONE repair
    (unchanged); ft-v2 (H-B1, common to every arm): at most TWO;
  * still invalid -> ``rejected``, the plan is NOT executed;
  * a transport failure is NOT repaired and not scored as a planner failure: ``infra_failure`` is set so it can be
    re-queued (preregistration s13). A refusal, truncation or empty reply goes through the repair like any invalid
    reply. ``first_ok`` (raw first-pass validity) and ``repaired`` are reported separately.
  * ft-v2 only: every attempt also records ``codes`` (the validator's error codes) and the result carries
    ``first_codes`` / ``codes_by_attempt`` / ``contract``; ft-v1 results are unchanged key for key.
  * ft-v2 only, feedback contract ``fb-1`` (FREEZE-GATES step 2, 2026-10-13; Astra FREEZE-GATES Q1): the problem text sent in a
    repair turn is ``cap_feedback(problems_text(errors))``: folded to ASCII (NFKD, combining marks dropped, any other non-ASCII
    character -> '?', control characters except newline -> ' ') and cut to at most PROBLEM_TEXT_CAP_CHARS characters INCLUDING the
    omission marker (whole lines kept while at least one fits; otherwise the first line is cut inside). Token bound: every character
    of the folded alphabet (95 printable ASCII + newline) is ONE token of the Gemma vocabulary (diag/freeze_gates/gemma_vocab_probe),
    so the text is <= 600 Gemma tokens; the API tokenizers get a conservative 2 tokens per character (measured max 1.02 over 16
    adversarial 600-char families, diag/freeze_gates/ascii_bound_probe). The original errors stay in ``attempts[k].validation`` (full
    audit); ``attempts[k].feedback`` records raw / sent sizes, hashes and whether folding or truncation happened. ft-v1: no cap.
"""
from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Callable, Optional

from .contract import MAX_REPAIRS_V2, PlannerContract, problems_text, schema_problems

MAX_REPAIRS = 1
#: ft-v2 repair-feedback contract (Astra FREEZE-GATES Q1); disclosed in the ledger next to treatment d5831abc..., pinned by gate 12.1
FEEDBACK_CONTRACT = "fb-1"
#: the problem text of one repair turn, folded to ASCII, is at most this many characters INCLUDING the omission marker
PROBLEM_TEXT_CAP_CHARS = 600
#: tokens per folded character, by tokenizer: gemma = proven (each folded character is one vocabulary token); the API tokenizers =
#: a conservative factor (measured max 1.02 tokens / char, Anthropic count_tokens, 16 adversarial families)
PROBLEM_TOKENS_PER_CHAR = {"gemma": 1.0, "anthropic": 2.0, "codex": 2.0}
OMIT_MARKER = "- (feedback cut: {n} more characters not shown)"
_FOLDED = frozenset(chr(o) for o in range(32, 127)) | {"\n"}


def problem_token_bound(tokenizer: str) -> int:
    """The token bound of one capped problem text under `tokenizer` (gemma / anthropic / codex / max)."""
    f = max(PROBLEM_TOKENS_PER_CHAR.values()) if tokenizer == "max" else PROBLEM_TOKENS_PER_CHAR[tokenizer]
    return int(math.ceil(f * PROBLEM_TEXT_CAP_CHARS))


def fold_ascii(text: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKD", text or ""):
        if ch in _FOLDED:
            out.append(ch)
        elif ord(ch) < 32 or ord(ch) == 127:
            out.append(" ")
        elif unicodedata.combining(ch):
            continue
        else:
            out.append("?")
    return "".join(out)


def cap_feedback(text: str, cap: int = PROBLEM_TEXT_CAP_CHARS):
    """-> (the text sent, audit info). fb-1: ASCII fold, then at most `cap` characters INCLUDING the omission marker."""
    raw = text or ""
    folded = fold_ascii(raw)
    info = {"contract": FEEDBACK_CONTRACT, "cap": int(cap), "chars_raw": len(raw), "chars_folded": len(folded),
            "folded": folded != raw, "truncated": False, "omitted_chars": 0,
            "sha_raw": hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()}   # a JSON \udXXX escape can leave a lone surrogate
    sent = folded
    if len(folded) > cap:
        reserve = len("\n" + OMIT_MARKER.format(n=len(folded)))        # omitted < len(folded): its digits always fit
        room = cap - reserve
        if room < 1:
            raise ValueError("feedback cap too small for the omission marker")
        head = folded[:room]
        cut = head.rfind("\n")
        if cut > 0 and folded[room] != "\n":                            # keep whole lines when at least one fits
            head = head[:cut]
        head = head.rstrip("\n")
        omitted = len(folded) - len(head)
        sent = head + "\n" + OMIT_MARKER.format(n=omitted)
        info.update(truncated=True, omitted_chars=omitted)
    if len(sent) > cap or not set(sent) <= _FOLDED:
        raise AssertionError("fb-1 invariant broken")
    info.update(chars_sent=len(sent), sha_sent=hashlib.sha256(sent.encode("utf-8")).hexdigest())
    return sent, info


def _default_validate(obj, types, holding, tools):
    import step4_gemma
    return step4_gemma.validate(obj, types, holding, tools)


def _default_validate_v2(obj, types, holding, tools, observed_types=None):
    from . import validator_v2
    return validator_v2.validate(obj, types, holding, tools, observed_types)


class PlannerRunner:
    def __init__(self, provider, contract: PlannerContract, validate: Optional[Callable] = None,
                 max_repairs: Optional[int] = None, budget: Optional[dict] = None):
        self.provider, self.contract = provider, contract
        self.v2 = getattr(contract, "version", "ft-v1") == "ft-v2"
        self.validate = validate or (_default_validate_v2 if self.v2 else _default_validate)
        self.max_repairs = int(max_repairs if max_repairs is not None else (MAX_REPAIRS_V2 if self.v2 else MAX_REPAIRS))
        self.budget = budget

    def _validate(self, obj, holding, observed_types):
        if self.v2 and observed_types is not None:
            return self.validate(obj, self.contract.types, holding, self.contract.tools, observed_types)
        return self.validate(obj, self.contract.types, holding, self.contract.tools)

    def _repair(self, messages, raw, problems):
        fn = getattr(self.contract, "repair_turn", None)
        return fn(messages, raw, problems) if fn else self.contract.repair_messages(messages, raw, problems)

    def plan(self, command: str, memory_summary=(), holding: Optional[str] = None, memory_block: Optional[str] = None,
             fewshot_block: Optional[str] = None, replicate: int = 0, observed_types=None) -> dict:
        messages = self.contract.messages(command, memory_summary, holding, memory_block, fewshot_block)
        attempts, infra = [], False
        for k in range(1 + self.max_repairs):
            res = self.provider.plan(messages, self.contract.schema, self.budget, replicate,
                                     stage="plan" if k == 0 else "repair")
            obj = res["parsed"]
            problems = []
            if res["error_kind"] == "transport":
                infra = True
            if obj is None:
                problems.append({"i": None, "reason": f"reply is not a JSON plan ({res['error'] or res['parse_note']})"})
                v = {"ok": False, "steps": [], "errors": problems, "notes": [], "normalised": []}
                if self.v2:
                    problems[0]["code"] = "E_NO_PLAN"
            else:
                shape = schema_problems(obj, self.contract.schema)
                if shape:
                    v = {"ok": False, "steps": [], "errors": [{"i": None, "reason": s} for s in shape[:5]],
                         "notes": [], "normalised": []}
                    if self.v2:
                        for e in v["errors"]:
                            e["code"] = "E_SCHEMA"
                else:
                    v = self._validate(obj, holding, observed_types)
                    if res["error"] and v["ok"]:                    # e.g. truncated yet parseable: still an error
                        v = {**v, "ok": False, "errors": [{"i": None, "reason": res["error"]}]}
                        if self.v2:
                            v["errors"][0]["code"] = "E_SCHEMA"
            att = {"k": k, "result": res, "validation": v}
            if self.v2:
                att["codes"] = [e.get("code") for e in v["errors"]]
            attempts.append(att)
            if v["ok"] or infra:
                break
            fb = problems_text(v["errors"])
            if self.v2:
                fb, att["feedback"] = cap_feedback(fb)
            messages = self._repair(messages, res["raw"], fb)
        last = attempts[-1]
        ok = last["validation"]["ok"]
        parsed = last["result"]["parsed"]
        out = {"ok": ok, "plan": last["validation"]["steps"] if ok else [],
               "say": parsed.get("say", "") if isinstance(parsed, dict) else "",
               "first_ok": attempts[0]["validation"]["ok"], "repaired": ok and len(attempts) > 1,
               "normalised": last["validation"].get("normalised", []) if ok else [],
               "notes": last["validation"].get("notes", []), "rejected": not ok and not infra,
               "infra_failure": infra, "attempts": attempts, "first_plan": attempts[0]["validation"]["steps"],
               "calls": len(attempts), "cached_calls": sum(1 for a in attempts if a["result"]["cached"]),
               "tokens_in": sum(a["result"]["tokens_in"] for a in attempts),
               "tokens_out": sum(a["result"]["tokens_out"] for a in attempts),
               "cost_usd": round(sum(a["result"]["cost_usd"] for a in attempts), 6),
               "secs": round(sum(a["result"]["latency_s"] or 0 for a in attempts), 3)}
        if self.v2:
            out.update({"contract": "ft-v2", "first_codes": attempts[0]["codes"],
                        "codes_by_attempt": [a["codes"] for a in attempts],
                        "unknown_types": last["validation"].get("unknown", [])})
        return out
