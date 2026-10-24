"""C2-e: the real-token counter the memory block's retention uses (final test, stage C3).

counter(text) = MAX over {Gemma (Ollama prompt_eval_count difference), Anthropic (/v1/messages/count_tokens, free)} of the
tokens `text` adds over a one-token baseline ("."), so counts are additive-ish and per-record accounting works. The planner-prompt
header (MEMORY_HEADER) is NOT inside each count; `header_cost` (max over tokenizers) is what `LongTermMemory.memory_block`
subtracts once from the level budget, which keeps "block + header <= budget" without charging the header to every record.
Results are cached on disk (text sha -> counts), so a re-run, a resume and every arm see the same numbers.

The Anthropic key is read only through `planners.keys.get_anthropic_key` (THESIS_ANTHROPIC_API_KEY), never printed or stored.
A missing key is NOT silently ignored: `allow_missing_anthropic=True` must be passed explicitly and is then recorded in
`self.tokenizers` (so a manifest can state that the retention was Gemma-only).
"""
from __future__ import annotations

import hashlib
import json
import os
import http.client
import threading
import time
import urllib.error
import urllib.request

#: DRESS-REHEARSAL 2026-10-14 (plumbing): a dropped connection / 429 / 5xx on a token count is retried (the count itself is unchanged;
#: a deterministic 4xx is raised at once). Delays in seconds between attempts; after the last one the error is raised as before.
RETRY_DELAYS = (2.0, 5.0, 15.0, 30.0)
_TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError)


def _transient(e) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, _TRANSIENT)


def post_json(req, timeout, delays=RETRY_DELAYS, sleep=time.sleep):
    """urlopen(req).read() parsed as JSON, retried on transient failures only (see RETRY_DELAYS)."""
    for i in range(len(delays) + 1):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:                                 # noqa: BLE001
            if i >= len(delays) or not _transient(e):
                raise
            sleep(delays[i])

OLLAMA_MODEL = "gemma4:e2b-it-q4_K_M"
ANTHROPIC_MODEL = "claude-sonnet-5-5"


class RealMaxCounter:
    def __init__(self, cache_path, header, ollama_endpoint="http://127.0.0.1:11434", ollama_model=OLLAMA_MODEL,
                 allow_missing_anthropic=False, timeout=300):
        self.cache_path, self.header = cache_path, header
        self.endpoint, self.model, self.timeout = ollama_endpoint.rstrip("/"), ollama_model, timeout
        self._lock = threading.Lock()
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, encoding="utf-8"))
            except (OSError, ValueError):
                self.cache = {}
        self.tokenizers = ["gemma"]
        self._key = None
        try:
            from planners import keys as K
            self._key = K.get_anthropic_key(None)
            self.tokenizers.append("anthropic")
        except Exception as e:                                  # PlannerSkip (no key)
            if not allow_missing_anthropic:
                raise RuntimeError("Anthropic key unavailable (THESIS_ANTHROPIC_API_KEY): the C2-e counter needs both tokenizers; "
                                   "pass allow_missing_anthropic=True to run Gemma-only (recorded)") from e
        self.n_calls = {"gemma": 0, "anthropic": 0}
        self._base = None

    # ---- raw counters
    def _gemma_raw(self, txt):
        body = {"model": self.model, "messages": [{"role": "user", "content": txt}], "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 1, "top_k": 1}}
        req = urllib.request.Request(self.endpoint + "/api/chat", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
        self.n_calls["gemma"] += 1
        return int(post_json(req, self.timeout)["prompt_eval_count"])

    def _anth_raw(self, txt):
        body = {"model": ANTHROPIC_MODEL, "messages": [{"role": "user", "content": txt}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", json.dumps(body).encode("utf-8"),
                                     {"x-api-key": self._key.reveal(), "anthropic-version": "2023-06-01", "content-type": "application/json"})
        self.n_calls["anthropic"] += 1
        return int(post_json(req, 60)["input_tokens"])

    def baseline(self):
        if self._base is None:
            self._base = {"gemma": self._cached(".", "gemma_raw", self._gemma_raw)}
            if self._key is not None:
                self._base["anthropic"] = self._cached(".", "anth_raw", self._anth_raw)
        return self._base

    def _cached(self, txt, field, fn):
        k = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]
        e = self.cache.setdefault(k, {})
        if field not in e:
            e[field] = fn(txt)
            e["chars"] = len(txt)
        return e[field]

    def save(self):
        with self._lock:
            tmp = self.cache_path + ".tmp"
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh)
            os.replace(tmp, self.cache_path)

    @property
    def header_cost(self) -> int:
        if getattr(self, "_hc", None) is None:
            self._hc = max(1, max(self.counts(self.header + "\n.").values()))
        return self._hc

    def counts(self, text) -> dict:
        if not text.strip():                       # the Anthropic endpoint refuses whitespace-only content; a separator is one token
            return {k: 1 for k in self.tokenizers}
        full = text
        b = self.baseline()
        out = {"gemma": self._cached(full, "gemma_raw", self._gemma_raw) - b["gemma"]}
        if self._key is not None:
            out["anthropic"] = self._cached(full, "anth_raw", self._anth_raw) - b["anthropic"]
        return out

    def __call__(self, text) -> int:
        if not text:
            return 0
        c = self.counts(text)
        return max(1, max(c.values()))
