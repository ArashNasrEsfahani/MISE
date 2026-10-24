"""PlannerProvider interface shared by every planner backend (final test, stage P1).

``plan(messages, schema, budget)`` returns one result dict:
  raw, parsed, tokens_in, tokens_out, cost_usd, latency_s, model_id, provider, decoding   (the contract)
plus: cached, stop_reason, error, error_kind ("" | "transport" | "length" | "refusal" | "stop" | "empty"),
prompt_sha256, cache_key, replicate, stage, extra (provider facts: wire request sha256, role mapping, served model),
parse_note, original_cost_usd / original_latency_s (on a cache replay: the cost and latency of the generation).

Template method: the base class owns the cache lookup, the cost reservation + settlement, timing, parsing and
redaction; subclasses implement ``_call`` (one round trip, using ``self.post`` so every transmission is reserved
and rate limited) and ``decoding``. A cache replay is never another observation: it costs 0 incremental USD.

Raises ``PlannerSkip`` (no key / no CLI: a clean skip, never a failure) and ``CostCeilingExceeded`` (abort; nothing
was sent). One provider instance is a sequential queue: do not share it between threads (the ledger itself is
multi-process safe).
"""
from __future__ import annotations

import abc
import hashlib
import json
import math
import time
import urllib.error
import urllib.request

#: statuses that mean "not processed, not billed": retried, reservation released
RETRY_UNBILLED = (408, 425, 429, 503, 529)
#: statuses that may have consumed tokens: retried, reservation kept OPEN (unresolved) until the run ends
RETRY_AMBIGUOUS = (500, 502, 504)


class PlannerSkip(Exception):
    """A planner cannot run here (missing key, CLI absent). Reported as a skip, never a crash or a failure."""


class PlannerError(Exception):
    pass


class CostCeilingExceeded(PlannerError):
    """The committed spend plus the reserve for the next transmission would pass a ceiling. Nothing was sent."""


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def messages_text(messages) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def estimate_tokens(text: str) -> int:
    """Pessimistic (chars/2: Persian, JSON and schema overhead tokenise worse than English prose). Used ONLY for
    the pre-call spend reserve, so the ceiling is approximate; the billed usage settles the real cost."""
    return int(math.ceil(len(text) / 2.0))


# ------------------------------------------------------------------------------------------------ transport
def http_post_json(url: str, headers: dict, body: dict, timeout: float):
    """-> (status, response headers dict, parsed JSON or text). HTTP error statuses are returned, not raised."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, hdrs, data = r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        status, hdrs, data = e.code, (dict(e.headers.items()) if e.headers else {}), e.read()
    text = data.decode("utf-8", "replace")
    try:
        return status, hdrs, json.loads(text)
    except ValueError:
        return status, hdrs, text


def http_get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class RateLimiter:
    def __init__(self, min_interval_s: float = 0.0, clock=time.monotonic, sleep=time.sleep):
        self.min_interval_s, self._clock, self._sleep, self._last = float(min_interval_s), clock, sleep, None

    def wait(self):
        if self._last is not None and self.min_interval_s > 0:
            gap = self._clock() - self._last
            if gap < self.min_interval_s:
                self._sleep(self.min_interval_s - gap)
        self._last = self._clock()


class TransportFailure(Exception):
    """Retries exhausted (or a non-retryable HTTP status). ``status`` is None for connection errors."""

    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


# ------------------------------------------------------------------------------------------------ the interface
class PlannerProvider(abc.ABC):
    provider = "base"
    exploratory = False
    role_mapping = "system+user chat roles"

    def __init__(self, model: str, cache=None, ledger=None, prices=None, transport=None, sleep=time.sleep,
                 clock=time.monotonic, min_interval_s: float = 0.0, max_retries: int = 3, backoff_s: float = 2.0,
                 timeout_s: float = 240.0):
        from . import cache as _cache, ledger as _ledger, prices as _prices
        self.model = model
        self.cache = cache if cache is not None else _cache.ResponseCache()
        self.ledger = ledger if ledger is not None else _ledger.CostLedger()
        self.prices = prices if prices is not None else _prices.PriceTable.load()
        self._transport = transport or http_post_json
        self._sleep, self.max_retries, self.backoff_s, self.timeout_s = sleep, int(max_retries), float(backoff_s), timeout_s
        self._limiter = RateLimiter(min_interval_s, clock, sleep)
        self._reserve_amt, self._open_rids, self._ok_rid, self._wire_sha = 0.0, [], None, ""

    # ---- subclass surface
    @abc.abstractmethod
    def decoding(self) -> dict:
        """Everything that changes the reply: frozen, recorded, part of the cache key."""

    @abc.abstractmethod
    def _call(self, messages, schema, budget) -> dict:
        """One logical round trip -> {raw, tokens_in, tokens_out, stop_reason, error, error_kind, extra,
        cost_usd?}. May raise PlannerSkip (before anything is sent) or TransportFailure."""

    def check_available(self):
        """Raise PlannerSkip when the provider cannot run. Called AFTER the cache lookup, so a replay of cached
        plans works without a key."""

    def max_output_tokens(self, budget) -> int:
        return int((budget or {}).get("max_tokens") or self.decoding().get("max_tokens") or 1024)

    def reserve_usd(self, messages, budget) -> float:
        out_tokens = int((budget or {}).get("reserve_output_tokens") or self.max_output_tokens(budget))
        return self.prices.cost(self.provider, self.model, estimate_tokens(messages_text(messages)), out_tokens)

    # ---- the public call
    def plan(self, messages, schema, budget=None, replicate: int = 0, use_cache: bool = True,
             stage: str = "plan") -> dict:
        from .contract import parse_reply, prompt_sha256
        from .keys import redact
        budget = dict(budget or {})
        dec = self.decoding()
        if budget.get("max_tokens"):                    # P3: a per-call output budget changes the reply -> cache key
            dec = {**dec, "max_tokens_override": int(budget["max_tokens"])}
        psha = prompt_sha256(messages)
        key = self.cache.key(self.provider, self.model, psha, dec, replicate, schema, stage)
        base = {"model_id": self.model, "provider": self.provider, "decoding": dec, "prompt_sha256": psha,
                "cache_key": key, "replicate": replicate, "stage": stage, "exploratory": self.exploratory}
        hit = self.cache.get(key) if use_cache else None
        if hit is not None:
            res = {**base, **{k: hit.get(k) for k in ("raw", "tokens_in", "tokens_out", "stop_reason", "error",
                                                      "error_kind", "extra")},
                   "latency_s": 0.0, "cost_usd": 0.0, "cached": True,
                   "original_cost_usd": hit.get("cost_usd", 0.0), "original_latency_s": hit.get("latency_s")}
            res["parsed"], res["parse_note"] = parse_reply(res["raw"] or "")
            self.ledger.book(self._entry(res, cached=True))
            return res
        self.check_available()
        self._reserve_amt, self._open_rids, self._ok_rid, self._wire_sha = \
            self.reserve_usd(messages, budget), [], None, ""
        t0 = time.time()
        try:
            out = self._call(messages, schema, budget)
        except TransportFailure as e:
            out = {"raw": "", "tokens_in": 0, "tokens_out": 0, "stop_reason": "", "extra": {"http_status": e.status},
                   "error": redact(f"transport: {e}")[:300], "error_kind": "transport"}
        except CostCeilingExceeded:
            raise
        except BaseException:
            self._release_all()
            raise
        latency = round(time.time() - t0, 3)
        raw = out.get("raw") or ""
        cost = out.get("cost_usd")
        if cost is None:
            cost = self.prices.cost(self.provider, self.model, out.get("tokens_in", 0), out.get("tokens_out", 0),
                                    actual=True)
        err = redact(out.get("error") or "")
        kind = out.get("error_kind") or ("empty" if not raw.strip() and not err else "")
        parsed, note = parse_reply(raw) if raw else (None, "empty reply")
        extra = dict(out.get("extra") or {})
        extra.update({"request_sha256": self._wire_sha, "role_mapping": self.role_mapping,
                      "unresolved_reservations": len([r for r in self._open_rids if r != self._ok_rid])})
        res = {**base, "raw": raw, "parsed": parsed, "parse_note": note, "tokens_in": int(out.get("tokens_in") or 0),
               "tokens_out": int(out.get("tokens_out") or 0), "cost_usd": round(float(cost), 6), "latency_s": latency,
               "stop_reason": out.get("stop_reason") or "", "error": err, "error_kind": kind, "cached": False,
               "extra": extra}
        self._settle(res)
        if use_cache and kind != "transport":
            self.cache.put(key, {k: res[k] for k in ("raw", "tokens_in", "tokens_out", "cost_usd", "latency_s",
                                                      "stop_reason", "error", "error_kind", "extra")},
                           {"provider": self.provider, "model": self.model, "prompt_sha256": psha,
                            "replicate": replicate, "stage": stage, "decoding": dec})
        return res

    # ---- reservations
    def begin_transmission(self) -> int:
        """Reserve the worst case for ONE transmission (re-reserved for every retransmission)."""
        rid = self.ledger.reserve(self.provider, self._reserve_amt)
        self._open_rids.append(rid)
        return rid

    def _release_all(self):
        for rid in self._open_rids:
            self.ledger.release(rid)
        self._open_rids = []

    def _settle(self, res):
        entry = self._entry(res, cached=False)
        if self._ok_rid is not None:
            self.ledger.settle(self._ok_rid, res["cost_usd"], entry)
        else:
            self.ledger.book(entry)
        # earlier ambiguous attempts stay open (counted at their reserve) -- they may have been billed

    @staticmethod
    def _entry(res, cached) -> dict:
        return {"provider": res["provider"], "model": res["model_id"], "prompt_sha256": res["prompt_sha256"],
                "replicate": res["replicate"], "stage": res.get("stage", "plan"), "tokens_in": res["tokens_in"],
                "tokens_out": res["tokens_out"], "cost_usd": res["cost_usd"], "cached": cached,
                "latency_s": res.get("latency_s"), "error_kind": res.get("error_kind") or "",
                "stop_reason": res.get("stop_reason") or ""}

    def post(self, url, headers, body, secrets=()):
        """POST with rate limit and up to ``max_retries`` retries (exponential backoff, honours Retry-After). Every
        transmission is reserved first; an unbilled rejection releases its reservation, an ambiguous failure
        (timeout, connection reset, 500/502/504) keeps it open. Returns parsed JSON; raises TransportFailure."""
        from .keys import redact
        self._wire_sha = sha256_json(body)
        last = None
        for k in range(self.max_retries + 1):
            rid = self.begin_transmission()
            self._limiter.wait()
            retry_after, ambiguous = None, True
            try:
                status, hdrs, data = self._transport(url, headers, body, self.timeout_s)
            except PlannerSkip:
                self.ledger.release(rid)
                raise
            except Exception as e:                                  # connection reset, timeout, DNS ...
                last = TransportFailure(redact(f"{type(e).__name__}: {e}", secrets)[:250])
            else:
                if 200 <= status < 300 and isinstance(data, dict):
                    self._ok_rid = rid
                    return data
                msg = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
                last = TransportFailure(redact(f"HTTP {status}: {msg[:200]}", secrets), status)
                ambiguous = status in RETRY_AMBIGUOUS
                if status not in RETRY_UNBILLED and status not in RETRY_AMBIGUOUS:
                    self.ledger.release(rid)                        # 4xx etc.: rejected, not billed, not retried
                    self._open_rids.remove(rid)
                    raise last
                try:
                    retry_after = float({str(a).lower(): b for a, b in (hdrs or {}).items()}.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
            if not ambiguous:
                self.ledger.release(rid)
                self._open_rids.remove(rid)
            if k < self.max_retries:
                self._sleep(retry_after if retry_after is not None else self.backoff_s * (2 ** k))
        raise last

    def __repr__(self):
        return f"<{type(self).__name__} {self.model}>"
