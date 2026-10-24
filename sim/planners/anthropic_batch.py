"""Anthropic Message Batches path for the plan-first planner (stage BATCH-A29, 2026-10-13). NEW code; the synchronous provider
(`anthropic_provider.py`) is not changed.

What stays identical to the synchronous path
  * the request BODY: `AnthropicPlanner.build_body` builds it (same model, system, messages, output_config.format + effort, thinking,
    max_tokens), so the wire sha256 of a batch request equals the sync one byte for byte (`request_sha256`);
  * the response CACHE key (`ResponseCache.key`, same decoding dict: the tier is NOT part of the decoding, because the body is the same);
    a batch answer is written to the same cache with the same manifest, so a synchronous resume replays it;
  * the REPAIR frames: the one `PlannerRunner` builds every repair turn. The batch driver re-runs `runner.plan` for each unit every round;
    known answers come back from the session store, the first unknown call raises `NeedCall` (its exact body + cache key). One round =
    one batch. So repair k of a cell is rendered by the same code from the same failed reply as in the sync path;
  * the reply parsing (`parse_message`: text blocks by type, usage, stop_reason mapping) mirrors `AnthropicPlanner._call` (the selftest
    checks the equivalence on the same message).

What differs (recorded on every result, `extra`):
  * billing: `cost_usd` is the BATCH price (BATCH_DISCOUNT x list, on input, output and cache reads); `list_cost_usd` is the list-price
    sensitivity; `billing_tier` = "batch";
  * latency: `latency_s` = the batch's wall time (submission -> ended, as observed by polling); there is no per-request latency;
  * custom_id: the API allows only ^[A-Za-z0-9_-]{1,64}$. A cell key ("ns|plan|arm|canon:item:lang|r0") is used verbatim when it fits,
    otherwise `ck_` + sha256(cell key)[:56]; the mapping is in every submission record (`batch_submitted`).

Result types (docs: errored / canceled / expired are not billed):
  succeeded -> parsed + billed; errored with error.type invalid_request_error -> not retried (= a sync 4xx: transport failure, `infra`);
  any other errored / canceled / expired -> resubmitted in a later round, at most `max_retries` times (= the sync rule for unbilled
  429/503/529), then a transport failure (`infra`, re-queued in a later session under the registered s13 rule).

Cost guard: every request is RESERVED in the ledger at its own pre-call reserve (batch price) BEFORE the batch is created (a refusal ->
nothing is sent); a succeeded result SETTLES its reservation at the billed batch cost; unbilled results RELEASE it. A batch whose results
never come (killed process) leaves its reservations OPEN and a `batch_submitted` record without `batch_collected`: the next start
re-attaches to that batch (never re-submits it).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request

from .anthropic_provider import API_VERSION, AnthropicPlanner
from .base import CostCeilingExceeded, PlannerSkip, TransportFailure, sha256_json

BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
#: Batch processing doc (https://platform.claude.com/docs/en/build-with-claude/batch-processing, read 2026-10-13): "All usage is charged at
#: 50% of the standard API prices"; Opus 5.5 batch input 2 / output 10 USD per MTok; caching discounts stack.
BATCH_DISCOUNT = 0.5
CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
NON_RETRYABLE_ERRORS = ("invalid_request_error", "invalid_request", "authentication_error", "permission_error", "not_found_error")
RETRY_HTTP = (408, 425, 429, 500, 502, 503, 504, 529)
UNBILLED_TYPES = ("errored", "canceled", "expired")


def custom_id_for(cell_key: str) -> str:
    """The cell key itself when the API accepts it, else a deterministic digest of it (recorded with the submission)."""
    if CUSTOM_ID_RE.fullmatch(cell_key or ""):
        return cell_key
    return "ck_" + hashlib.sha256((cell_key or "").encode("utf-8")).hexdigest()[:56]


class NeedCall(Exception):
    """Raised by BatchPlanner.plan for a call whose answer is not known yet: the batch driver collects it and submits it."""

    def __init__(self, key, body, messages, stage, replicate, reserve_usd, manifest):
        super().__init__("need batch call")
        self.key, self.body, self.messages, self.stage, self.replicate = key, body, messages, stage, replicate
        self.reserve_usd, self.manifest = reserve_usd, manifest


class BatchResultMissing(RuntimeError):
    """An ended batch returned no result line for some request: it is UNRESOLVED (it may have been billed). Its reservation stays open,
    nothing is retried, and the run stops for reconciliation (Astra A29 6)."""

    def __init__(self, batch_id, custom_ids):
        super().__init__(f"batch {batch_id}: {len(custom_ids)} request(s) without a result line")
        self.batch_id, self.custom_ids = batch_id, list(custom_ids)


class BatchSubmitAmbiguous(TransportFailure):
    """Creating the batch failed in a way that may still have created it (timeout / 5xx after send): the reservations stay OPEN."""


# ================================================================================================ HTTP client
def http_request(method: str, url: str, headers: dict, body=None, timeout: float = 120.0):
    """-> (status, headers, text). HTTP error statuses are returned, not raised."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (dict(e.headers.items()) if e.headers else {}), e.read().decode("utf-8", "replace")


class BatchClient:
    """Raw HTTPS over the injectable `transport` (the repo's planner layer is stdlib-first, like AnthropicPlanner). The key is read at call
    time from `key_getter` (-> keys.Secret) and only revealed into the header; error texts are redacted."""

    def __init__(self, key_getter, transport=http_request, url: str = BATCH_URL, sleep=time.sleep, retries: int = 3,
                 backoff_s: float = 5.0, timeout_s: float = 120.0):
        self.key_getter, self._t, self.url, self._sleep = key_getter, transport, url, sleep
        self.retries, self.backoff_s, self.timeout_s = int(retries), float(backoff_s), float(timeout_s)

    def _headers(self):
        key = self.key_getter()
        return key, {"x-api-key": key.reveal(), "anthropic-version": API_VERSION, "content-type": "application/json"}

    def _do(self, method, url, body=None, idempotent=True):
        from .keys import redact
        key, hdr = self._headers()
        last = None
        for k in range(self.retries + 1):
            try:
                st, hh, txt = self._t(method, url, hdr, body, self.timeout_s)
            except Exception as e:                                   # noqa: BLE001  connection reset / timeout
                msg = redact(f"{type(e).__name__}: {e}", (key,))[:250]
                if not idempotent:
                    raise BatchSubmitAmbiguous(f"batch create: {msg} (the batch may exist)") from None
                last = TransportFailure(msg)
            else:
                if 200 <= st < 300:
                    return txt
                last = TransportFailure(redact(f"HTTP {st}: {txt[:200]}", (key,)), st)
                if st not in RETRY_HTTP:
                    raise last
                if not idempotent and st in (500, 502, 504):
                    raise BatchSubmitAmbiguous(str(last), st)
            if k < self.retries:
                self._sleep(self.backoff_s * (2 ** k))
        raise last

    def create(self, requests: list) -> dict:
        return json.loads(self._do("POST", self.url, {"requests": requests}, idempotent=False))

    def retrieve(self, batch_id: str) -> dict:
        return json.loads(self._do("GET", f"{self.url}/{batch_id}"))

    def cancel(self, batch_id: str) -> dict:
        return json.loads(self._do("POST", f"{self.url}/{batch_id}/cancel", {}))

    def results(self, batch: dict) -> list:
        url = batch.get("results_url") or f"{self.url}/{batch['id']}/results"
        txt = self._do("GET", url)
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


# ================================================================================================ parsing
def parse_message(data: dict, body: dict, prices, model: str, discount: float = BATCH_DISCOUNT) -> dict:
    """A succeeded batch result's `message` -> the provider's `_call` result shape (same rules as AnthropicPlanner._call); cost at the
    batch price, `list_cost_usd` at list price."""
    text = "".join(b.get("text", "") for b in (data.get("content") or []) if isinstance(b, dict) and b.get("type") == "text")
    u = data.get("usage") or {}
    tin = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
    cache_read = u.get("cache_read_input_tokens") or 0
    tout = u.get("output_tokens") or 0
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
    lst = prices.cost("anthropic", model, tin, tout)
    if cache_read:
        lst += cache_read * prices.info("anthropic", model).get("cache_read", 0.0) / 1e6
    return {"raw": text, "tokens_in": tin + cache_read, "tokens_out": tout, "stop_reason": stop, "error": err, "error_kind": kind,
            "cost_usd": round(lst * discount, 8), "list_cost_usd": round(lst, 8), "served_model": data.get("model"),
            "message_id": data.get("id"), "cache_read_tokens": cache_read,
            "cache_creation_tokens": u.get("cache_creation_input_tokens") or 0, "service_tier": u.get("service_tier")}


# ================================================================================================ the provider
class BatchPlanner(AnthropicPlanner):
    """AnthropicPlanner whose `plan()` never transmits: it serves answers the batch driver collected this session (`fresh`, billed,
    cached=False), replays cache hits exactly like the base class, and raises NeedCall for anything else."""

    tier = "batch"

    def __init__(self, model: str, discount: float = BATCH_DISCOUNT, **kw):
        super().__init__(model, **kw)
        self.discount = float(discount)
        self.fresh: dict = {}             # cache key -> response dict (billed by this session's batches; served as NOT cached)
        self.failures: dict = {}          # cache key -> unbilled failures so far (expired / canceled / retryable errored)

    def reserve_usd(self, messages, budget) -> float:
        return self.discount * super().reserve_usd(messages, budget)

    def _key(self, messages, schema, budget, replicate, stage):
        from .contract import prompt_sha256
        budget = dict(budget or {})
        dec = self.decoding()
        if budget.get("max_tokens"):
            dec = {**dec, "max_tokens_override": int(budget["max_tokens"])}
        psha = prompt_sha256(messages)
        return self.cache.key(self.provider, self.model, psha, dec, replicate, schema, stage), psha, dec

    def plan(self, messages, schema, budget=None, replicate: int = 0, use_cache: bool = True, stage: str = "plan") -> dict:
        from .contract import parse_reply
        key, psha, dec = self._key(messages, schema, budget, replicate, stage)
        if key in self.fresh:
            r = self.fresh[key]
            res = {"model_id": self.model, "provider": self.provider, "decoding": dec, "prompt_sha256": psha, "cache_key": key,
                   "replicate": replicate, "stage": stage, "exploratory": self.exploratory, "cached": False,
                   **{k: r.get(k) for k in ("raw", "tokens_in", "tokens_out", "cost_usd", "latency_s", "stop_reason", "error",
                                            "error_kind", "extra")}}
            res["raw"] = res["raw"] or ""
            res["parsed"], res["parse_note"] = parse_reply(res["raw"]) if res["raw"] else (None, "empty reply")
            return res
        if use_cache and self.cache.get(key) is not None:
            return super().plan(messages, schema, budget, replicate, use_cache, stage)       # a plain replay (0 USD, booked)
        body = self.build_body(messages, schema, budget)
        raise NeedCall(key, body, messages, stage, replicate, self.reserve_usd(messages, budget),
                       {"provider": self.provider, "model": self.model, "prompt_sha256": psha, "replicate": replicate,
                        "stage": stage, "decoding": dec})

    def _call(self, messages, schema, budget):                   # never reached: plan() above does not transmit
        raise RuntimeError("BatchPlanner does not make synchronous calls")


# ================================================================================================ the driver
class BatchDriver:
    """Runs units through rounds of batches.

    unit = {"id": label (cell key), "provider": BatchPlanner, "run": callable() -> result (calls runner.plan / Plan._call_cell),
            "scope": ledger scope, "cell_tag": ledger cell tag}
    `ledger` must support reserve / settle / release and `cell_tag` / `use(scope)` (ScopedLedger), or plain CostLedger (no scope).
    `journal(event, **fields)` persists submission / collection records (resume). `now`, `sleep` injectable."""

    def __init__(self, client, ledger, journal=None, poll_s: float = 30.0, wait_timeout_s: float = 26 * 3600.0, sleep=time.sleep,
                 now=time.time, max_retries: int = 3, out=None):
        self.client, self.ledger, self.journal = client, ledger, journal or (lambda event, **kw: None)
        self.poll_s, self.wait_timeout_s, self._sleep, self._now = float(poll_s), float(wait_timeout_s), sleep, now
        self.max_retries = int(max_retries)
        self.out = out or (lambda m: None)
        self.stats = {"batches": 0, "requests": 0, "succeeded": 0, "errored": 0, "expired": 0, "canceled": 0, "resubmitted": 0,
                      "billed_usd": 0.0, "list_usd": 0.0, "batch_wall_s": []}

    # ---- one batch
    def submit(self, items: list, round_no: int, tag: str = "") -> dict:
        """items = [(unit, NeedCall)]. Reserves every request first (all or nothing), then creates ONE batch."""
        rids = []
        try:
            for u, n in items:
                self._tag(u)
                with self._scope(u):
                    rids.append(self.ledger.reserve("anthropic", n.reserve_usd))
        except CostCeilingExceeded:
            for r in rids:
                self.ledger.release(r)
            raise
        finally:
            self._tag(None)
        reqs, idx = [], []
        for (u, n), rid in zip(items, rids):
            cid = custom_id_for(u["id"])
            reqs.append({"custom_id": cid, "params": n.body})
            idx.append({"custom_id": cid, "cell": u["id"], "cell_tag": u.get("cell_tag") or u["id"], "cache_key": n.key, "stage": n.stage, "rid": rid,
                        "request_sha256": sha256_json(n.body), "reserve_usd": round(n.reserve_usd, 6), "manifest": n.manifest,
                        "max_tokens": n.body.get("max_tokens")})
        if len({r["custom_id"] for r in reqs}) != len(reqs):
            for r in rids:
                self.ledger.release(r)
            raise RuntimeError("two requests of one batch share a custom_id (one call per cell per round)")
        t0 = self._now()
        try:
            b = self.client.create(reqs)
        except BatchSubmitAmbiguous as e:
            self.journal("batch_submit_ambiguous", round=round_no, tag=tag, items=idx, detail=str(e)[:300])
            raise
        except BaseException:
            for r in rids:
                self.ledger.release(r)
            raise
        rec = {"batch_id": b["id"], "round": round_no, "tag": tag, "submitted_ts": t0, "items": idx}
        self.journal("batch_submitted", **rec)
        self.stats["batches"] += 1
        self.stats["requests"] += len(reqs)
        self.out(f"  batch {b['id']} round {round_no}: {len(reqs)} request(s) submitted")
        return rec

    def wait(self, batch_id: str, submitted_ts: float) -> dict:
        """Polls until the batch has ended. Past `wait_timeout_s` the batch is cancelled ONCE (its unprocessed requests end as
        `canceled`, unbilled) and polling continues until the API reports `ended`."""
        cancelled = False
        while True:
            b = self.client.retrieve(batch_id)
            if b.get("processing_status") == "ended":
                return b
            if not cancelled and self._now() - submitted_ts > self.wait_timeout_s:
                self.client.cancel(batch_id)
                cancelled = True
                self.journal("batch_cancel_timeout", batch_id=batch_id, waited_s=round(self._now() - submitted_ts, 1))
            self._sleep(self.poll_s)

    def collect(self, rec: dict, providers_by_key: dict, units_by_cell: dict | None = None) -> dict:
        """Settles / releases every reservation of a submitted batch and stores each answer in the provider's session store and the
        response cache. -> {custom_id: result type}."""
        b = self.wait(rec["batch_id"], rec["submitted_ts"])
        wall = round(max(0.0, self._now() - rec["submitted_ts"]), 3)
        self.stats["batch_wall_s"].append(wall)
        res = {r["custom_id"]: r.get("result") or {} for r in self.client.results(b)}
        types, missing = {}, []
        for it in rec["items"]:
            prov = providers_by_key.get(it["cache_key"])
            if it["custom_id"] not in res:                         # unresolved: keep the reservation OPEN, never retry
                types[it["custom_id"]] = "missing"
                missing.append(it["custom_id"])
                continue
            r = res[it["custom_id"]]
            t = r.get("type", "errored")
            types[it["custom_id"]] = t
            if hasattr(self.ledger, "cell_tag"):
                self.ledger.cell_tag = it.get("cell_tag") or it["cell"]
            try:
                if t == "succeeded" and prov is not None:
                    self._succeeded(prov, it, r["message"], wall, rec["batch_id"])
                else:
                    self.ledger.release(it["rid"])                 # errored / canceled / expired: not billed
                    self.stats["errored" if t == "errored" else t] = self.stats.get("errored" if t == "errored" else t, 0) + 1
                    if prov is not None:
                        self._unbilled(prov, it, r, t, wall, rec["batch_id"])
            finally:
                if hasattr(self.ledger, "cell_tag"):
                    self.ledger.cell_tag = None
        self.journal("batch_collected", batch_id=rec["batch_id"], round=rec["round"], tag=rec.get("tag", ""), wall_s=wall,
                     counts=b.get("request_counts"), types=types)
        if missing:
            self.journal("batch_result_missing", batch_id=rec["batch_id"], custom_ids=missing)
            raise BatchResultMissing(rec["batch_id"], missing)
        return types

    def _succeeded(self, prov, it, msg, wall, batch_id):
        from .keys import redact
        p = parse_message(msg, {"max_tokens": (it.get("max_tokens") or prov.profile["max_tokens"])}, prov.prices, prov.model,
                          prov.discount)
        extra = {"served_model": p["served_model"], "request_id": p["message_id"], "cache_read_tokens": p["cache_read_tokens"],
                 "cache_creation_tokens": p["cache_creation_tokens"], "billing_tier": "batch", "batch_id": batch_id,
                 "custom_id": it["custom_id"], "list_cost_usd": p["list_cost_usd"], "batch_discount": prov.discount,
                 "batch_wall_s": wall, "request_sha256": it["request_sha256"], "role_mapping": prov.role_mapping,
                 "unresolved_reservations": 0, "service_tier": p["service_tier"]}
        resp = {"raw": p["raw"], "tokens_in": int(p["tokens_in"]), "tokens_out": int(p["tokens_out"]),
                "cost_usd": round(float(p["cost_usd"]), 6), "latency_s": wall, "stop_reason": p["stop_reason"],
                "error": redact(p["error"] or ""), "error_kind": p["error_kind"], "extra": extra}
        man = dict(it.get("manifest") or {})
        prov.cache.put(it["cache_key"], resp, man or {"provider": prov.provider, "model": prov.model, "stage": it["stage"]})
        prov.fresh[it["cache_key"]] = resp
        entry = {"provider": prov.provider, "model": prov.model, "prompt_sha256": man.get("prompt_sha256"),
                 "replicate": man.get("replicate"), "stage": it["stage"], "tokens_in": resp["tokens_in"], "tokens_out": resp["tokens_out"],
                 "cost_usd": resp["cost_usd"], "cached": False, "latency_s": wall, "error_kind": resp["error_kind"],
                 "stop_reason": resp["stop_reason"], "billing_tier": "batch", "list_cost_usd": p["list_cost_usd"], "batch_id": batch_id,
                 "custom_id": it["custom_id"]}
        self.ledger.settle(it["rid"], resp["cost_usd"], entry)
        self.stats["succeeded"] += 1
        self.stats["billed_usd"] = round(self.stats["billed_usd"] + resp["cost_usd"], 6)
        self.stats["list_usd"] = round(self.stats["list_usd"] + p["list_cost_usd"], 6)

    def _unbilled(self, prov, it, r, t, wall, batch_id):
        etype = ((r.get("error") or {}).get("error") or {}).get("type") or (r.get("error") or {}).get("type") or ""
        n = prov.failures.get(it["cache_key"], 0) + 1
        prov.failures[it["cache_key"]] = n
        final = (t == "errored" and etype in NON_RETRYABLE_ERRORS) or n > self.max_retries
        if not final:
            self.stats["resubmitted"] += 1
            return                                                  # the next round asks again (NeedCall) -> resubmitted
        prov.fresh[it["cache_key"]] = {                             # a transport failure: never cached, the runner marks it infra
            "raw": "", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_s": wall, "stop_reason": "",
            "error": f"transport: batch result {t}" + (f" ({etype})" if etype else "") + f" after {n} submission(s)",
            "error_kind": "transport",
            "extra": {"http_status": None, "billing_tier": "batch", "batch_id": batch_id, "custom_id": it["custom_id"],
                      "batch_result": t, "batch_error_type": etype, "unresolved_reservations": 0}}

    # ---- rounds
    def run(self, units: list, tag: str = "") -> dict:
        """Round loop: each round re-runs every unfinished unit; NeedCalls are batched; ends when every unit returned. -> {id: result}."""
        done, rnd = {}, 0
        while True:
            need = []
            for u in units:
                if u["id"] in done:
                    continue
                try:
                    done[u["id"]] = u["run"]()
                except NeedCall as n:
                    need.append((u, n))
            if not need:
                return done
            rec = self.submit(need, rnd, tag)
            self.collect(rec, {n.key: u["provider"] for u, n in need})
            rnd += 1

    def _tag(self, u):
        if hasattr(self.ledger, "cell_tag"):
            self.ledger.cell_tag = (u or {}).get("cell_tag") if u else None

    def _scope(self, u):
        import contextlib
        if hasattr(self.ledger, "use") and u.get("scope"):
            return self.ledger.use(u["scope"])
        return contextlib.nullcontext()

