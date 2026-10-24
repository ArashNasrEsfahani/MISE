"""Pluggable LLM/VLM provider interface for the agent's "brain".

Decouples planning / active-perception / embedding from any single vendor so experiments can swap backends
(thesis requirement: keep the LLM provider behind an interface). The default backend is a local **Ollama**
server reached over its HTTP API with a stdlib-urllib transport (no extra dependency); a hosted **Claude**
backend is stubbed for later wiring.

Every provider exposes the same surface:
  - chat(model, messages, **opts) -> str                  text completion / planning
  - chat_vision(model, messages, images, **opts) -> str   multimodal (VLM active perception); optional
  - embed(texts) -> list[list[float]]                     embeddings for RAG; optional
  - list_models() -> list[str]                            models the backend offers (best-effort)

Stay stdlib-first: import any heavy dependency (numpy, anthropic, …) LAZILY inside the method that needs it,
NEVER at module top, so this module imports under a bare-stdlib Python (e.g. the Windows dev box).
"""
from __future__ import annotations

import abc
import json
import os
import urllib.request


class LLMProvider(abc.ABC):
    """Abstract LLM/VLM backend. Subclasses must implement at least :meth:`chat`."""

    @abc.abstractmethod
    def chat(self, model: str, messages: list, **opts) -> str:
        """Return the assistant's text reply for a chat `messages` list."""
        raise NotImplementedError

    def chat_vision(self, model: str, messages: list, images: list, **opts) -> str:
        """Multimodal chat with `images` attached to the conversation. Default: unsupported."""
        raise NotImplementedError(f"{type(self).__name__} does not support vision")

    def embed(self, texts) -> list:
        """Embed a string or list of strings -> list of float vectors. Default: unsupported."""
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")

    def list_models(self) -> list:
        """Names of models the backend offers (best-effort; default empty)."""
        return []


class OllamaProvider(LLMProvider):
    """Local Ollama backend over its HTTP API (stdlib urllib — no extra dependency).

    Ported verbatim from the original `agent_llm.Ollama` (chat via /api/chat, list_models via /api/tags) and
    extended with :meth:`embed` via /api/embeddings. The GUI runs inside WSL, so when Ollama lives on Windows
    the endpoint must be the Windows host (see `agent_llm.gateway_endpoint`), and Ollama must be started with
    `OLLAMA_HOST=0.0.0.0` so it accepts connections from the WSL NAT network.
    """

    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint.rstrip("/")

    def list_models(self, timeout: float = 5.0) -> list:
        data = self._get("/api/tags", timeout)
        return [m["name"] for m in data.get("models", [])]

    def chat(self, model: str, messages: list, timeout: float = 180.0,
             fmt: str | None = "json", options: dict | None = None, think: bool | None = None, **opts) -> str:
        payload = {"model": model, "messages": messages, "stream": False}
        if fmt:
            payload["format"] = fmt
        if options:
            payload["options"] = options
        if think is not None:                 # top-level, not an option (thinking models, e.g. Gemma 4)
            payload["think"] = bool(think)
        data = self._post("/api/chat", payload, timeout)
        #: completion facts of the last call (done_reason "length" = the output budget ran out)
        self.last_meta = {k: data.get(k) for k in ("done_reason", "eval_count", "prompt_eval_count")}
        self.last_meta["thinking_chars"] = len(data.get("message", {}).get("thinking") or "")
        return data.get("message", {}).get("content", "")

    def embed(self, texts, model: str = "nomic-embed-text", timeout: float = 60.0) -> list:
        """Embed one string or a list of strings via /api/embeddings -> list of float vectors."""
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            data = self._post("/api/embeddings", {"model": model, "prompt": t}, timeout)
            out.append(data.get("embedding", []))
        return out

    # -- transport (stdlib urllib) --
    def _get(self, path: str, timeout: float):
        req = urllib.request.Request(self.endpoint + path)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _post(self, path: str, payload: dict, timeout: float):
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())


class ClaudeProvider(LLMProvider):
    """Anthropic Claude backend — STUB (not yet wired).

    Reads `ANTHROPIC_API_KEY` from the environment but intentionally adds no `anthropic` dependency yet; every
    call raises NotImplementedError. Kept as a placeholder so the architecture can later compare a hosted VLM
    against local Ollama without changing call sites.
    """

    def __init__(self, api_key: str | None = None, endpoint: str | None = None, **kw):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.endpoint = endpoint

    def chat(self, model: str, messages: list, **opts) -> str:
        raise NotImplementedError("Claude provider not yet wired")


def make_provider(name: str = "ollama", endpoint: str = "http://localhost:11434", **kw) -> LLMProvider:
    """Construct an LLM provider by `name` (case-insensitive): 'ollama' (default) or 'claude'/'anthropic'."""
    key = (name or "ollama").strip().lower()
    if key in ("ollama", "local"):
        return OllamaProvider(endpoint, **kw)
    if key in ("claude", "anthropic"):
        return ClaudeProvider(endpoint=endpoint, **kw)
    raise ValueError(f"unknown LLM provider: {name!r} (expected 'ollama' or 'claude')")
