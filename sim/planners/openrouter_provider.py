"""OpenRouter provider (OpenAI-compatible chat endpoint). Exploratory: one cheap model on a small paired sample.

The key is read AT CALL TIME from the env file named by OPENROUTER_ENV_FILE (default D:\\KareOne\\RAG Paper\\.env,
variable OPENROUTER_API_KEY) and never copied. A hard spend cap (``hard_cap_usd``, default 1.00 of the ~1.4 USD left)
is enforced on top of the ledger. Fallbacks and transforms are disabled and the served provider/model is recorded
(an arm must be one named model on one named endpoint).
"""
from __future__ import annotations

from .base import CostCeilingExceeded, PlannerProvider
from .keys import get_openrouter_key

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterPlanner(PlannerProvider):
    provider = "openrouter"
    exploratory = True

    def __init__(self, model: str, max_tokens: int = 2000, hard_cap_usd: float = 1.00, env=None,
                 upstream: str | None = None, api_url: str = API_URL, **kw):
        super().__init__(model, **kw)
        self.max_tokens, self.hard_cap_usd, self._env, self.upstream, self.api_url = \
            int(max_tokens), float(hard_cap_usd), env, upstream, api_url

    def decoding(self) -> dict:
        return {"temperature": 0, "max_tokens": self.max_tokens, "response_format": "json_object",
                "allow_fallbacks": False, "require_parameters": True, "transforms": [], "upstream": self.upstream}

    def check_available(self):
        get_openrouter_key(self._env)

    def reserve_usd(self, messages, budget) -> float:
        r = super().reserve_usd(messages, budget)
        if self.ledger.spent(self.provider) + r > self.hard_cap_usd + 1e-12:
            raise CostCeilingExceeded(f"OpenRouter hard cap {self.hard_cap_usd:.2f} USD would be passed; nothing sent")
        return r

    def build_body(self, messages, schema=None, budget=None) -> dict:
        prov = {"allow_fallbacks": False, "require_parameters": True}
        if self.upstream:
            prov["order"] = [self.upstream]
        return {"model": self.model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "temperature": 0, "max_tokens": int((budget or {}).get("max_tokens") or self.max_tokens),
                "response_format": {"type": "json_object"}, "provider": prov, "transforms": [],
                "usage": {"include": True}}

    def _call(self, messages, schema, budget) -> dict:
        key = get_openrouter_key(self._env)
        body = self.build_body(messages, schema, budget)
        headers = {"Authorization": "Bearer " + key.reveal(), "Content-Type": "application/json"}
        data = self.post(self.api_url, headers, body, secrets=(key,))
        ch = (data.get("choices") or [{}])[0]
        raw = (ch.get("message") or {}).get("content") or ""
        u = data.get("usage") or {}
        tin, tout = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        fin = ch.get("finish_reason") or ""
        err = kind = ""
        if fin == "length":
            err, kind = f"output budget exhausted (finish_reason=length, max_tokens={body['max_tokens']})", "length"
        elif fin in ("content_filter", "error"):
            err, kind = f"finish_reason={fin}", "refusal" if fin == "content_filter" else "stop"
        elif not raw.strip():
            err, kind = "empty reply", "empty"
        cost = u.get("cost")
        if cost is None:
            cost = self.prices.cost(self.provider, self.model, tin, tout)
        return {"raw": raw, "tokens_in": tin, "tokens_out": tout, "stop_reason": fin, "error": err, "error_kind": kind,
                "cost_usd": float(cost),
                "extra": {"served_model": data.get("model"), "served_provider": data.get("provider"),
                          "request_id": data.get("id"), "cost_from_response": u.get("cost") is not None}}
