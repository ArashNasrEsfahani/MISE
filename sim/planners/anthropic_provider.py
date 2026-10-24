"""Anthropic Messages API provider (raw HTTPS over the injectable transport: the repo's llm layer is stdlib-first).

Frozen per-model configuration (preregistration s3.1) -- equal numbers do not make sampling equivalent, so each
model's settings are recorded, not claimed equal:
  claude-opus-5-5            adaptive thinking (cannot be disabled), effort medium, no sampling params, max 8000
  claude-sonnet-5-5          adaptive thinking, effort high, no sampling params, max 8000
  claude-haiku-4-5-20251001  no thinking, temperature 0, max 2000
Structured output: ``output_config.format`` = json_schema (the PLAN_SCHEMA shape; transmitted form recorded by hash).
No server-side fallback is ever requested. ``stop_reason`` max_tokens / refusal / context-exceeded are distinct planner
errors; thinking tokens are billed inside ``output_tokens``. The key is read from ANTHROPIC_API_KEY only.
"""
from __future__ import annotations

import copy
import os

from .base import PlannerProvider, PlannerSkip, sha256_json
from .keys import Secret, get_anthropic_key, redact

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

PROFILES = {
    "claude-opus-5-5": {"thinking": {"type": "adaptive"}, "effort": "medium", "temperature": None, "max_tokens": 8000},
    "claude-sonnet-5-5": {"thinking": {"type": "adaptive"}, "effort": "high", "temperature": None, "max_tokens": 8000},
    "claude-haiku-4-5-20251001": {"thinking": None, "effort": None, "temperature": 0, "max_tokens": 2000},
}
_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
                "multipleOf", "format", "minItems", "maxItems")


def to_anthropic_schema(schema: dict) -> dict:
    """The transmitted form of a JSON schema: ``additionalProperties: false`` on objects and type lists as
    ``anyOf`` (semantically identical). A keyword that cannot be expressed raises ValueError -- never silently
    weakened for one arm (Astra)."""
    s = copy.deepcopy(schema)

    def walk(n, path="$"):
        if isinstance(n, dict):
            for k in _UNSUPPORTED:
                if k in n:
                    raise ValueError(f"{path}: schema keyword {k!r} is not supported by structured output")
            t = n.get("type")
            if isinstance(t, list):
                rest = {k: v for k, v in n.items() if k != "type"}
                variants = []
                for x in t:
                    v = {"type": x}
                    if x == "object" or x == "array":
                        v.update({k: copy.deepcopy(val) for k, val in rest.items() if k in ("properties", "required", "items")})
                    variants.append(v)
                n.clear()
                n["anyOf"] = variants
                for v in variants:
                    walk(v, path)
                return
            if t == "object":
                n["additionalProperties"] = False
            for k, v in list(n.items()):
                walk(v, f"{path}.{k}")
        elif isinstance(n, list):
            for i, v in enumerate(n):
                walk(v, f"{path}[{i}]")

    walk(s)
    return s


class AnthropicPlanner(PlannerProvider):
    provider = "anthropic"
    role_mapping = "system -> top-level `system`; user/assistant -> messages"

    def __init__(self, model: str, effort: str | None = "__profile__", max_tokens: int | None = None, env=None,
                 api_url: str = API_URL, **kw):
        if model not in PROFILES:
            raise ValueError(f"unknown Anthropic model {model!r}; frozen profiles: {', '.join(PROFILES)}")
        super().__init__(model, **kw)
        self.profile = dict(PROFILES[model])
        if effort != "__profile__":
            self.profile["effort"] = effort
        if max_tokens:
            self.profile["max_tokens"] = int(max_tokens)
        self._env, self.api_url = env, api_url

    def decoding(self) -> dict:
        return {**self.profile, "structured_output": "output_config.format:json_schema", "fallbacks": "off",
                "api_version": API_VERSION}

    def check_available(self):
        get_anthropic_key(self._env)                                # PlannerSkip when unset

    def build_body(self, messages, schema, budget=None) -> dict:
        from .contract import split_system
        system, turns = split_system(messages)
        p = self.profile
        body = {"model": self.model, "max_tokens": int((budget or {}).get("max_tokens") or p["max_tokens"]),
                "system": system, "messages": [{"role": m["role"], "content": m["content"]} for m in turns]}
        out_cfg = {"format": {"type": "json_schema", "schema": to_anthropic_schema(schema)}}
        if p.get("effort"):
            out_cfg["effort"] = p["effort"]
        body["output_config"] = out_cfg
        if p.get("thinking"):
            body["thinking"] = dict(p["thinking"])
        if p.get("temperature") is not None:
            body["temperature"] = p["temperature"]
        return body

    def _call(self, messages, schema, budget) -> dict:
        key = get_anthropic_key(self._env)
        body = self.build_body(messages, schema, budget)
        headers = {"x-api-key": key.reveal(), "anthropic-version": API_VERSION, "content-type": "application/json"}
        data = self.post(self.api_url, headers, body, secrets=(key,))
        # parse blocks by type, never content[0] (a thinking block can come first)
        text = "".join(b.get("text", "") for b in (data.get("content") or []) if isinstance(b, dict) and b.get("type") == "text")
        u = data.get("usage") or {}
        tin = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
        cache_read = u.get("cache_read_input_tokens") or 0
        tout = u.get("output_tokens") or 0                         # includes thinking tokens
        stop = data.get("stop_reason") or ""
        err = kind = ""
        if stop == "refusal":
            cat = (data.get("stop_details") or {}).get("category")
            err, kind = f"refusal (category={cat})", "refusal"
        elif stop == "max_tokens":
            err, kind = f"output budget exhausted (stop_reason=max_tokens, max_tokens={body['max_tokens']})", "length"
        elif stop == "model_context_window_exceeded":
            err, kind = "context window exceeded", "length"
        elif stop not in ("end_turn", "stop_sequence"):
            err, kind = f"unexpected stop_reason {stop!r}", "stop"
        elif not text.strip():
            err, kind = "empty reply", "empty"
        cost = self.prices.cost(self.provider, self.model, tin, tout)
        if cache_read:
            cost += cache_read * self.prices.info(self.provider, self.model).get("cache_read", 0.0) / 1e6
        return {"raw": text, "tokens_in": tin + cache_read, "tokens_out": tout, "stop_reason": stop, "error": err,
                "error_kind": kind, "cost_usd": cost,
                "extra": {"served_model": data.get("model"), "request_id": data.get("id"),
                          "cache_read_tokens": cache_read, "schema_canonical_sha256": sha256_json(schema),
                          "schema_transmitted_sha256": sha256_json(body["output_config"]["format"]["schema"])}}
