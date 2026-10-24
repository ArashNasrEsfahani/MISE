"""Cost ledger with a global ceiling (and optional per-provider ceilings) that ABORTS.

SQLite (stdlib) so that reservations are atomic across processes: a call is refused when
``settled + open reservations + the new reservation`` would pass a ceiling. Every transmission reserves its worst
case first (``PlannerProvider.begin_transmission``); a settled call replaces its reservation with the billed cost; a
clean rejection releases it; an ambiguous failure (timeout after send, 5xx) stays OPEN and keeps counting, because
it may have been billed. The ledger holds provider, model, prompt sha256, token counts and dollars -- never a prompt
or a key. Every event is also appended to ``<db>.jsonl`` for audit. Journal mode is the SQLite default (not WAL):
the store may sit on a /mnt DrvFs path.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time

from .base import CostCeilingExceeded

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "final_test", "ledger.db")
SMOKE_CEILINGS = {"anthropic": 1.0, "openrouter": 0.10}             # P1 smoke caps (preregistration section 10)


class CostLedger:
    def __init__(self, path: str | None = None, global_ceiling_usd: float | None = None,
                 provider_ceilings: dict | None = None, audit: bool = True):
        self.path = path or DEFAULT_DB
        self.global_ceiling = global_ceiling_usd
        self.provider_ceilings = dict(provider_ceilings or {})
        self.audit = audit
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, "
                      "provider TEXT, state TEXT, reserved REAL, actual REAL, entry TEXT)")

    @classmethod
    def smoke(cls, path: str | None = None, global_ceiling_usd: float = 1.10) -> "CostLedger":
        return cls(path, global_ceiling_usd, SMOKE_CEILINGS)

    @contextlib.contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            yield c
        finally:
            c.close()

    def _committed(self, c, provider=None) -> float:
        q = "SELECT COALESCE(SUM(CASE WHEN state='open' THEN reserved WHEN state='settled' THEN actual ELSE 0 END),0)" \
            " FROM ledger"
        if provider:
            return c.execute(q + " WHERE provider=?", (provider,)).fetchone()[0]
        return c.execute(q).fetchone()[0]

    def spent(self, provider: str | None = None) -> float:
        """Committed spend: billed + still-open (possibly billed) reservations."""
        with self._lock, self._conn() as c:
            return round(self._committed(c, provider), 6)

    def billed(self, provider: str | None = None) -> float:
        with self._lock, self._conn() as c:
            q = "SELECT COALESCE(SUM(actual),0) FROM ledger WHERE state='settled'"
            r = c.execute(q + (" AND provider=?" if provider else ""), (provider,) if provider else ()).fetchone()
            return round(r[0], 6)

    def reserve(self, provider: str, usd: float) -> int:
        with self._lock, self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                tot, ptot = self._committed(c), self._committed(c, provider)
                pc = self.provider_ceilings.get(provider)
                if self.global_ceiling is not None and tot + usd > self.global_ceiling + 1e-12:
                    raise CostCeilingExceeded(f"global ceiling {self.global_ceiling:.2f} USD: committed {tot:.4f} "
                                              f"+ reserve {usd:.4f} for {provider}; nothing sent")
                if pc is not None and ptot + usd > pc + 1e-12:
                    raise CostCeilingExceeded(f"{provider} ceiling {pc:.2f} USD: committed {ptot:.4f} + reserve "
                                              f"{usd:.4f}; nothing sent")
                cur = c.execute("INSERT INTO ledger (ts, provider, state, reserved) VALUES (?,?,?,?)",
                                (_now(), provider, "open", float(usd)))
                rid = cur.lastrowid
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        self._audit({"event": "reserve", "id": rid, "provider": provider, "usd": round(usd, 6)})
        return rid

    def settle(self, rid: int, actual_usd: float, entry: dict):
        with self._lock, self._conn() as c:
            c.execute("UPDATE ledger SET state='settled', actual=?, entry=? WHERE id=?",
                      (float(actual_usd), json.dumps(entry, ensure_ascii=False), rid))
        self._audit({"event": "settle", "id": rid, **entry})

    def release(self, rid: int):
        with self._lock, self._conn() as c:
            n = c.execute("UPDATE ledger SET state='released' WHERE id=? AND state='open'", (rid,)).rowcount
        if n:
            self._audit({"event": "release", "id": rid})

    def book(self, entry: dict):
        """A call with no open reservation: a cache replay (0 USD) or a failure that sent nothing billable."""
        state = "cache" if entry.get("cached") else "settled"
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO ledger (ts, provider, state, reserved, actual, entry) VALUES (?,?,?,?,?,?)",
                      (_now(), entry["provider"], state, 0.0, float(entry.get("cost_usd") or 0.0),
                       json.dumps(entry, ensure_ascii=False)))
        self._audit({"event": "book", **entry})

    def rows(self) -> list:
        with self._lock, self._conn() as c:
            return [dict(zip(("id", "ts", "provider", "state", "reserved", "actual", "entry"), r))
                    for r in c.execute("SELECT id, ts, provider, state, reserved, actual, entry FROM ledger")]

    def _audit(self, rec: dict):
        if not self.audit:
            return
        try:
            with open(self.path + ".jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": _now(), **rec}, ensure_ascii=False) + "\n")
        except OSError:
            pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
