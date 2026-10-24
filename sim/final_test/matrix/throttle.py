"""Plan-first (PF-safety-test): the shared rate-limit throttle of the parallel planning phase.

One `Throttle` per provider FAMILY (anthropic, codex, gemma). It limits TRANSMISSIONS, not cells: every initial transmission, every
transport retry and every repair call of a provider goes through the transport (or CLI runner) wrapper below, so a provider that is already
inside its own retry loop cannot send through a cooldown. Two different numbers are tracked and reported:
  * active CELLS   - the worker threads that currently hold a cell of this family (bounded by the pool, `Plan` counts them);
  * active TRANSMISSIONS - requests on the wire right now (bounded by `limit`, which the throttle lowers under pressure).

Behaviour
  * `acquire()` blocks until the cooldown is over AND `active < limit`; it re-checks the stop flag (an abort refuses a new transmission);
  * a rate-limit answer (HTTP 429 / 503 / 529, or a CLI failure whose output says "rate limit") sets a pool-wide cooldown
    (Retry-After if given, else `base * 2^(n-1)` capped, with a small jitter), HALVES `limit` (never below 1) and resets the clean streak;
  * after `restore_after` clean calls `limit` grows by one, up to the configured cap;
  * the clock, the sleep and the random source are injectable, so the selftest runs on a fake clock.

Python 3.10 syntax. No key, no prompt text and no response text ever passes through here except the (already redacted by the provider)
status and header values.
"""
from __future__ import annotations

import random
import re
import threading
import time

RATE_STATUSES = (429, 503, 529)                    # = planners.base.RETRY_UNBILLED minus 408/425 (a timeout is not a rate limit)
RATE_TEXT = re.compile(r"rate.?limit|too many requests|usage limit|quota|\b429\b|overloaded", re.I)


class Aborted(Exception):
    """Raised inside a wrapper when the pool has been told to abort: a new transmission is refused (nothing is sent)."""


class Throttle:
    def __init__(self, name: str, cap: int, *, clock=time.monotonic, sleep=time.sleep, base_cooldown_s: float = 2.0,
                 max_cooldown_s: float = 120.0, jitter: float = 0.2, restore_after: int = 20, rng=None, stop_check=None,
                 poll_s: float = 0.01):
        self.name, self.cap = name, max(1, int(cap))
        self.limit = self.cap
        self._clock, self._sleep, self.base, self.max_cd = clock, sleep, float(base_cooldown_s), float(max_cooldown_s)
        self.jitter, self.restore_after, self._rng = float(jitter), int(restore_after), rng or random.Random(0)
        self.stop_check, self.poll_s = stop_check, poll_s
        self._lock = threading.Lock()
        self.cooldown_until = 0.0
        self.active = 0                              # transmissions on the wire now
        self.peak = 0                                # most transmissions on the wire at once
        self.cells = 0                               # cells of this family in progress (bumped by cell())
        self.peak_cells = 0
        self.n_sent = 0                              # transmissions started
        self.n_rate_limited = 0
        self.n_cooldowns = 0
        self.min_limit = self.cap
        self.consecutive = 0                         # rate limits in a row
        self.clean = 0
        self.waited_s = 0.0
        self.sent_at = []                            # clock stamps of every transmission (storm check; bounded below)
        self.events = []                             # [(clock, 'limit', new_limit, cooldown)] for the report

    # ------------------------------------------------------------------ cells (reporting only)
    def cell_enter(self):
        with self._lock:
            self.cells += 1
            self.peak_cells = max(self.peak_cells, self.cells)

    def cell_leave(self):
        with self._lock:
            self.cells -= 1

    # ------------------------------------------------------------------ transmissions
    def acquire(self):
        t0 = self._clock()
        while True:
            if self.stop_check is not None and self.stop_check():
                raise Aborted(self.name)
            with self._lock:
                now = self._clock()
                if now >= self.cooldown_until and self.active < self.limit:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    self.n_sent += 1
                    self.sent_at.append(now)
                    if len(self.sent_at) > 5000:
                        del self.sent_at[:1000]
                    self.waited_s += max(0.0, now - t0)
                    return
                wait = (self.cooldown_until - now) if now < self.cooldown_until else self.poll_s
            self._sleep(max(self.poll_s, min(wait, 1.0)))

    def release(self):
        with self._lock:
            self.active = max(0, self.active - 1)

    def ok(self):
        with self._lock:
            self.consecutive = 0
            self.clean += 1
            if self.clean >= self.restore_after and self.limit < self.cap:
                self.limit += 1
                self.clean = 0
                self.events.append((self._clock(), "restore", self.limit, 0.0))

    def rate_limited(self, retry_after=None):
        with self._lock:
            self.n_rate_limited += 1
            self.consecutive += 1
            self.clean = 0
            try:
                ra = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError):
                ra = None
            delay = ra if ra is not None and ra >= 0 else self.base * (2 ** min(self.consecutive - 1, 6))
            delay = min(self.max_cd, delay) * (1.0 + self.jitter * self._rng.random())
            now = self._clock()
            if now + delay > self.cooldown_until:
                self.cooldown_until = now + delay
            self.n_cooldowns += 1
            self.limit = max(1, self.limit // 2)
            self.min_limit = min(self.min_limit, self.limit)
            self.events.append((now, "limit", self.limit, round(delay, 3)))

    # ------------------------------------------------------------------ wrappers
    def wrap_transport(self, inner):
        """transport(url, headers, body, timeout) -> (status, headers, data). Used for the HTTP providers (Anthropic, Ollama)."""
        def transport(url, headers, body, timeout):
            self.acquire()
            try:
                status, hdrs, data = inner(url, headers, body, timeout)
            except BaseException:
                self.release()
                raise
            self.release()
            if status in RATE_STATUSES:
                low = {str(k).lower(): v for k, v in (hdrs or {}).items()}
                self.rate_limited(low.get("retry-after"))
            elif 200 <= int(status) < 300:
                self.ok()
            return status, hdrs, data
        transport.throttle = self
        return transport

    def wrap_runner(self, inner):
        """runner(cmd, prompt, cwd, env, timeout) -> CompletedProcess-like. Used for the Codex CLI (a `--version` probe is not a transmission)."""
        def runner(cmd, prompt, cwd, env, timeout):
            if "--version" in cmd:
                return inner(cmd, prompt, cwd, env, timeout)
            self.acquire()
            try:
                r = inner(cmd, prompt, cwd, env, timeout)
            except BaseException:
                self.release()
                raise
            self.release()
            text = f"{getattr(r, 'stderr', '') or ''}\n{getattr(r, 'stdout', '') or ''}"
            if getattr(r, "returncode", 0) != 0 and RATE_TEXT.search(text):
                self.rate_limited(None)
            elif getattr(r, "returncode", 0) == 0:
                self.ok()
            return r
        runner.throttle = self
        return runner

    # ------------------------------------------------------------------ report
    def max_rate(self, window_s: float = 1.0) -> int:
        """Most transmissions started inside any window of `window_s` seconds (the storm indicator)."""
        with self._lock:
            ts = sorted(self.sent_at)
        best, j = 0, 0
        for i, t in enumerate(ts):
            while ts[j] < t - window_s:
                j += 1
            best = max(best, i - j + 1)
        return best

    def report(self) -> dict:
        with self._lock:
            return {"family": self.name, "cap": self.cap, "limit_now": self.limit, "min_limit": self.min_limit,
                    "peak_transmissions": self.peak, "peak_cells": self.peak_cells, "sent": self.n_sent,
                    "rate_limited": self.n_rate_limited, "cooldowns": self.n_cooldowns, "waited_s": round(self.waited_s, 3)}
