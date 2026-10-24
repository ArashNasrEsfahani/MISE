"""Ollama provider (Gemma 4 E2B): the step-4 behaviour -- temperature 0, seed 7, top_k 1, num_ctx 8192, num_predict
768, think=False, JSON-Schema ``format`` -- over the same /api/chat payload ``llm.OllamaProvider.chat`` sends."""
from __future__ import annotations

from .base import PlannerProvider, TransportFailure, http_get_json

MODEL = "gemma4:e2b-it-q4_K_M"
OPTIONS = {"temperature": 0.0, "seed": 7, "num_ctx": 8192, "num_predict": 768, "top_k": 1}
THINK = False


class OllamaPlanner(PlannerProvider):
    provider = "ollama"

    def __init__(self, model: str = MODEL, endpoint: str = "http://127.0.0.1:11434", options: dict | None = None,
                 think: bool | None = THINK, strict_schema: bool = False, **kw):
        super().__init__(model, **kw)
        self.endpoint = endpoint.rstrip("/")
        self.options = dict(OPTIONS if options is None else options)
        self.think = think
        #: P3 (Astra): optional strict transform (additionalProperties:false, type lists -> anyOf) = the form the
        #: Anthropic / Codex arms receive. OFF by default (= step-4 behaviour); switching it on is an amendment that
        #: must be settled on dev items before practice.
        self.strict_schema = bool(strict_schema)

    def decoding(self) -> dict:
        d = {"options": dict(self.options), "think": self.think, "stream": False, "format": "json-schema",
             "max_tokens": self.options.get("num_predict")}
        if self.strict_schema:
            d["format"] = "json-schema-strict"
        return d

    def _call(self, messages, schema, budget) -> dict:
        fmt = schema
        if self.strict_schema:
            from .anthropic_provider import to_anthropic_schema
            fmt = to_anthropic_schema(schema)
        payload = {"model": self.model, "messages": messages, "stream": False, "format": fmt,
                   "options": dict(self.options)}
        if self.think is not None:
            payload["think"] = bool(self.think)
        data = self.post(self.endpoint + "/api/chat", {"Content-Type": "application/json"}, payload)
        msg = data.get("message") or {}
        raw = msg.get("content") or ""
        done = data.get("done_reason") or ""
        err = kind = ""
        if done == "length":
            err, kind = f"output budget exhausted (done_reason=length, num_predict={self.options.get('num_predict')})", "length"
        elif not raw.strip():
            err, kind = "empty reply", "empty"
        return {"raw": raw, "tokens_in": data.get("prompt_eval_count") or 0, "tokens_out": data.get("eval_count") or 0,
                "stop_reason": done, "error": err, "error_kind": kind,
                "extra": {"thinking_chars": len(msg.get("thinking") or ""), "total_duration_ns": data.get("total_duration")}}

    def environment(self) -> dict:
        """Best-effort facts for freeze.json: Ollama version and the model digest (never raises)."""
        out = {"endpoint_host": self.endpoint.split("//")[-1], "model": self.model}
        try:
            out["ollama_version"] = http_get_json(self.endpoint + "/api/version").get("version")
        except Exception as e:
            out["ollama_version"] = f"unavailable ({type(e).__name__})"
        try:
            for m in http_get_json(self.endpoint + "/api/tags").get("models", []):
                if m.get("name") == self.model:
                    out["digest"] = (m.get("digest") or "")[:12]
                    out["size_bytes"] = m.get("size")
        except Exception:
            pass
        return out
