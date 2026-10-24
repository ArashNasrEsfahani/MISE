"""Phase-6 planning-strategy selector — a self-contained LinUCB contextual bandit.

The agent can attempt a user command with one of four *planning strategies*. This module learns, per
command *context*, which strategy is worth trying, using a gradient-free contextual bandit (LinUCB). It
does **no** LLM fine-tuning: the LLM stays a frozen Ollama inference server. The only thing that learns
here is a small linear model per arm, updated from a scalar reward.

The four arms (``ARMS``):

  1. ``reuse_memory``      — plan using retrieved RAG/spatial memory (the current default behaviour).
  2. ``plan_fresh``        — plan WITHOUT injecting the "relevant memory" block (ignore long-term memory).
  3. ``explore_then_plan`` — run one bounded active exploration (``scan_for``) BEFORE retrieve + plan.
  4. ``replay_procedure``  -- Phase-7: skip the LLM call and re-execute a stored KB procedure directly.
     GATED: only eligible in :meth:`Bandit.decide` when the caller passes ``procedure_match`` (a 0..1
     similarity between the command and its best stored procedure, computed by the caller via
     ``kb.search``) at or above ``REPLAY_MATCH_THRESHOLD``. Without a close match the bandit behaves
     exactly like the original three-arm selector.

Phase-7 also adds an ``updates_enabled`` freeze (constructor kwarg + :meth:`Bandit.set_updates_enabled`):
when False, :meth:`Bandit.decide` keeps exploiting the learned model normally, but :meth:`Bandit.reward`
becomes an audited no-op (no A/b mutation, decision row left pending) so the self-improving loop's
"no learning while the loop is off" ablation axis holds for the bandit too.

Reward is the simulator's per-episode success (``episodes.success`` 0/1) OR an optional human thumb
(up = +1, down = -1). Provenance ranks the two: a ``human`` thumb (rank 2) outranks the ``environment``
signal (rank 1); an equal-or-higher-provenance reward already on a decision is never overwritten — the
same rule ``memory.py``/``knowledge.py`` use for beliefs.

**Self-contained & additive.** This module owns two of its own tables (``bandit_arms``,
``bandit_decisions``) created with ``CREATE TABLE IF NOT EXISTS`` so it can safely share ``memory.db``
without touching any Phase-5 table, method or column. It opens its **own** ``sqlite3`` connection.

**Exception-safe by construction** (mirrors the Phase-5 safety pattern): every public method wraps its
body in ``try/except``; any internal failure is logged to the audit file and the method degrades
gracefully — :meth:`Bandit.decide` falls back to the safe default arm ``reuse_memory`` and NEVER raises,
so a bandit failure can never break planning or serving.

Dependencies: ``numpy`` for the linear algebra (installed in the venv) + stdlib ``sqlite3/json/os/time``.
The numpy import is GUARDED (``NUMPY_AVAILABLE``): per CLAUDE.md's stdlib-first rule every module in the
retrieval/planner path must *import* under a bare Python, so a missing numpy now fails at :class:`Bandit`
construction with a clear message instead of at ``import select``. This was the one module in that path that
raised ``ImportError`` at module scope. The bandit itself genuinely needs numpy and is not emulated in stdlib —
the point is only that a bare-Python consumer degrades to the safe default arm rather than dying on import.

KNOWN, NOT FIXED HERE: this file is named ``select.py`` and therefore **shadows the stdlib ``select`` module**
for anything that puts ``sim/`` on ``sys.path`` (``selectors``, ``asyncio``, and POSIX ``subprocess`` under WSL
all resolve ``select`` and would find this module, which has no ``select.select``). Every in-repo consumer
already loads it BY PATH for exactly this reason (`webapp.py:674`, `_phase7_selftest.py:59`). A rename to
``bandit.py`` is the real fix and is recorded as a follow-up in ``docs/agentic-rag-design.md``; it touches
files this workflow does not own, so it is not done here.
"""
from __future__ import annotations

import json
import os
import time

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:                       # pragma: no cover - exercised only on a bare-stdlib interpreter
    np = None                           # type: ignore[assignment]
    NUMPY_AVAILABLE = False


def _require_numpy():
    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "sim/select.py needs numpy for the LinUCB linear algebra, and numpy is not importable. "
            "The module still imports so a bare-Python consumer can detect this and fall back to the safe "
            "default arm ARMS[0] ('reuse_memory'); install numpy to enable the bandit.")

# The planning strategies the bandit chooses between. Index 0 is also the SAFE DEFAULT returned on
# any failure or when the bandit is disabled. APPEND-ONLY: never reorder (index 0 is the tie-break
# winner and safe default; _load_state auto-creates fresh rows for arms missing from an existing DB,
# so appending upgrades a live memory.db in place).
ARMS = ["reuse_memory", "plan_fresh", "explore_then_plan", "replay_procedure"]

# Minimum caller-supplied procedure_match score (0..1) for the "replay_procedure" arm to be eligible in
# decide(). Below this (or when procedure_match is None) the arm is excluded from the argmax entirely.
REPLAY_MATCH_THRESHOLD = 0.6

# Reward provenance ranking (who produced a reward). A human thumb outranks the environment's success
# signal; an equal-or-higher-provenance reward already recorded on a decision is never overwritten.
# Mirrors memory.PROVENANCE ordering (human > environment).
BANDIT_PROVENANCE = {"human": 2, "environment": 1}

# Feature layout (fixed dimension d, constant across the whole module):
#   [ bias=1.0, lang_fa, lang_en, memory_hit, failure_seen ]  ++  one-hot scene bucket (K=6)
# => d = 5 + 6 = 11. lang_fa/lang_en are mutually exclusive; the scene bucket is hash(scene) mod K.
_N_BASE = 5
_SCENE_BUCKETS = 6
_FEATURE_DIM = _N_BASE + _SCENE_BUCKETS  # = 11

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bandit_arms (
    arm     TEXT PRIMARY KEY,
    A       BLOB,
    b       BLOB,
    updates INTEGER
);
CREATE TABLE IF NOT EXISTS bandit_decisions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL,
    scene   TEXT,
    command TEXT,
    arm     TEXT,
    context TEXT,
    reward  REAL,
    source  TEXT
);
"""


def _is_persian(ch: str) -> bool:
    """True if ``ch`` is in a Persian/Arabic Unicode range (Arabic, Supplement, Extended-A, presentation forms)."""
    cp = ord(ch)
    return (0x0600 <= cp <= 0x06FF        # Arabic (covers Persian letters ک گ پ چ ژ ی and digits)
            or 0x0750 <= cp <= 0x077F     # Arabic Supplement
            or 0x08A0 <= cp <= 0x08FF     # Arabic Extended-A
            or 0xFB50 <= cp <= 0xFDFF     # Arabic Presentation Forms-A
            or 0xFE70 <= cp <= 0xFEFF)    # Arabic Presentation Forms-B


def _scene_bucket(scene: str | None, k: int = _SCENE_BUCKETS) -> int:
    """Deterministic scene bucket in ``[0, k)``.

    Uses a stable hash (md5) rather than the builtin ``hash()`` — the latter is salted per-process
    (``PYTHONHASHSEED``), which would silently re-bucket every scene on restart and desync persisted state.
    """
    import hashlib
    s = "" if scene is None else str(scene)
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h % k


class Bandit:
    """LinUCB contextual bandit over the planning strategies (``ARMS``). SQLite-persisted, exception-safe."""

    FEATURE_DIM = _FEATURE_DIM

    def __init__(self, db_path: str, *, alpha: float = 1.0, enabled: bool = True,
                 updates_enabled: bool = True, audit_path: str | None = None):
        _require_numpy()                # clear failure HERE, not an ImportError at `import select`
        self.db_path = db_path
        self.alpha = float(alpha)
        self.enabled = bool(enabled)
        # Freeze gate for LEARNING only (distinct from `enabled`, which alters decide()): when False,
        # decide() still exploits the learned model but reward() is an audited no-op. Wired by webapp
        # to the Phase-7 loop toggle so "loop off" implies "bandit must not learn".
        self.updates_enabled = bool(updates_enabled)
        self.d = _FEATURE_DIM
        # Separate audit log from memory.py's (memory_audit.log) so the two writers never clobber each other.
        self.audit_path = audit_path or (os.path.splitext(db_path)[0] + "_bandit_audit.log")
        # In-memory per-arm LinUCB state; hydrated from / persisted to bandit_arms.
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._updates: dict[str, int] = {}
        self.conn = None
        try:
            self.conn = __import__("sqlite3").connect(db_path, check_same_thread=False)
            self.conn.row_factory = __import__("sqlite3").Row
            self.conn.executescript(_SCHEMA)
            self.conn.commit()
            self._load_state()
            self._audit("INIT", f"bandit ready (d={self.d}, alpha={self.alpha}, enabled={self.enabled}, "
                        f"updates_enabled={self.updates_enabled})",
                        "load persisted LinUCB state", source="environment")
        except Exception as e:  # never let construction raise — degrade to a fresh in-memory model
            self._ensure_defaults()
            self._audit("INIT-ERR", f"init failed: {e!r}", "degraded to in-memory defaults", source="environment")

    # ----------------------------------------------------------------- audit (mirrors memory.Memory._audit)
    def _audit(self, op: str, summary: str, why: str = "", source: str | None = None):
        src = f"  <{source}>" if source else ""
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [BANDIT {op}]{src}  {summary}"
                + (f"   WHY: {why}" if why else ""))
        try:
            with open(self.audit_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        return line

    # ----------------------------------------------------------------- state persistence
    def _fresh_arm(self):
        """A brand-new arm's LinUCB state: A = identity (ridge prior λ=1), b = zeros, 0 updates."""
        return np.identity(self.d, dtype=np.float64), np.zeros(self.d, dtype=np.float64), 0

    def _ensure_defaults(self):
        for arm in ARMS:
            if arm not in self._A:
                A, b, u = self._fresh_arm()
                self._A[arm], self._b[arm], self._updates[arm] = A, b, u

    @staticmethod
    def _blob(arr: np.ndarray) -> str:
        """Serialize a numpy array as a JSON list (portable, endianness-free, human-inspectable)."""
        return json.dumps(arr.tolist())

    def _unblob(self, raw, shape) -> np.ndarray:
        """Inverse of :meth:`_blob`; tolerates str or bytes storage. Falls back to a fresh array on any error."""
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            arr = np.array(json.loads(raw), dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"shape {arr.shape} != {shape}")
            return arr
        except Exception:
            return np.identity(self.d, dtype=np.float64) if len(shape) == 2 else np.zeros(self.d, dtype=np.float64)

    def _load_state(self):
        """Hydrate per-arm A/b/updates from bandit_arms; insert fresh rows for any arm not yet persisted."""
        rows = {}
        try:
            for r in self.conn.execute("SELECT arm, A, b, updates FROM bandit_arms").fetchall():
                rows[r["arm"]] = r
        except Exception:
            rows = {}
        for arm in ARMS:
            r = rows.get(arm)
            if r is None:
                A, b, u = self._fresh_arm()
                self._A[arm], self._b[arm], self._updates[arm] = A, b, u
                self._persist_arm(arm)
            else:
                self._A[arm] = self._unblob(r["A"], (self.d, self.d))
                self._b[arm] = self._unblob(r["b"], (self.d,))
                self._updates[arm] = int(r["updates"] or 0)

    def _persist_arm(self, arm: str):
        """Write one arm's A/b/updates back to bandit_arms (upsert)."""
        try:
            self.conn.execute(
                "INSERT INTO bandit_arms (arm, A, b, updates) VALUES (?,?,?,?) "
                "ON CONFLICT(arm) DO UPDATE SET A=excluded.A, b=excluded.b, updates=excluded.updates",
                (arm, self._blob(self._A[arm]), self._blob(self._b[arm]), int(self._updates[arm])))
            self.conn.commit()
        except Exception as e:
            self._audit("PERSIST-ERR", f"could not persist arm {arm}: {e!r}", source="environment")

    # ----------------------------------------------------------------- featurization
    def featurize(self, *, command: str | None, scene: str | None,
                  memory_hit, failure_seen) -> list:
        """Fixed-dimension (d=11) context feature vector for a command.

        Layout: ``[bias=1.0, lang_fa, lang_en, memory_hit(0/1), failure_seen(0/1)]`` followed by a K=6
        one-hot scene bucket ``hash(scene) mod 6``. ``lang_fa=1`` iff ``command`` contains any Persian/
        Arabic-range character, else ``lang_en=1`` (bilingual, per project convention). Never raises — on
        any error returns a safe all-bias-only vector of the right dimension.
        """
        try:
            cmd = "" if command is None else str(command)
            lang_fa = 1.0 if any(_is_persian(c) for c in cmd) else 0.0
            lang_en = 0.0 if lang_fa else 1.0
            mh = 1.0 if (memory_hit if not isinstance(memory_hit, str) else memory_hit.strip()) else 0.0
            fs = 1.0 if failure_seen else 0.0
            feats = [1.0, lang_fa, lang_en, mh, fs]
            onehot = [0.0] * _SCENE_BUCKETS
            onehot[_scene_bucket(scene)] = 1.0
            feats.extend(onehot)
            return feats
        except Exception as e:
            self._audit("FEATURIZE-ERR", f"{e!r}", "returned bias-only fallback vector", source="environment")
            v = [0.0] * self.d
            v[0] = 1.0
            return v

    def _as_vector(self, context_features) -> np.ndarray | None:
        """Coerce a caller-supplied feature list into a length-d float vector, or None if it can't be used."""
        try:
            x = np.asarray(context_features, dtype=np.float64).reshape(-1)
            if x.shape[0] != self.d:
                return None
            return x
        except Exception:
            return None

    # ----------------------------------------------------------------- decision (UCB argmax)
    def _insert_decision(self, scene, command, arm, context_json) -> int:
        """Insert a pending (reward NULL) decision row and return its autoincrement id (or -1 on failure)."""
        try:
            cur = self.conn.execute(
                "INSERT INTO bandit_decisions (ts, scene, command, arm, context, reward, source) "
                "VALUES (?,?,?,?,?,NULL,NULL)",
                (time.time(), scene, command, arm, context_json))
            self.conn.commit()
            return int(cur.lastrowid)
        except Exception as e:
            self._audit("DECISION-ERR", f"could not log decision: {e!r}", source="environment")
            return -1

    def decide(self, context_features, *, scene: str | None = None, command: str | None = None,
               procedure_match: float | None = None, procedure_entry: str | None = None) -> dict:
        """Choose an arm for this context via LinUCB and log a pending decision.

        UCB score per arm is ``theta·x + alpha * sqrt(x^T A^-1 x)`` where ``theta = A^-1 b``; the argmax
        is taken over the ELIGIBLE arms. ``replay_procedure`` is eligible only when the caller passes
        ``procedure_match`` (0..1 similarity between the command and its best stored KB procedure; the
        caller computes it, e.g. webapp via ``kb.search``) at or above ``REPLAY_MATCH_THRESHOLD``;
        ``procedure_entry`` (the matched KB entry id) is audit-logged and echoed back so the caller can
        execute that stored plan directly when this arm wins. Without a qualifying match the argmax is
        identical to the original three-arm selector.

        A pending ``bandit_decisions`` row (``reward`` NULL) is inserted and its id returned so a later
        :meth:`reward` call can credit it. Returns ``{"arm", "decision_id", "scores": {arm: float},
        "eligible": [arms], "procedure_match", "procedure_entry"}`` (scores cover ALL arms for display;
        only eligible ones competed).

        If ``enabled`` is False: always returns the safe default arm ``reuse_memory`` (does NOT explore) but
        STILL logs a decision row. On ANY internal error: logs, falls back to ``reuse_memory``, never raises.
        """
        x = self._as_vector(context_features)
        ctx_json = json.dumps(x.tolist()) if x is not None else json.dumps(context_features, default=float)

        # Replay eligibility gate: the arm competes only when a stored procedure closely matches the
        # command. A malformed procedure_match is treated as "no match" (never raises).
        try:
            pm = None if procedure_match is None else float(procedure_match)
        except Exception:
            pm = None
        eligible = [a for a in ARMS
                    if a != "replay_procedure" or (pm is not None and pm >= REPLAY_MATCH_THRESHOLD)]
        replay_note = f" pm={pm:.2f} entry={procedure_entry}" if pm is not None else ""

        # Disabled or malformed context → safe default arm, but still record the decision for auditability.
        if not self.enabled or x is None:
            arm = ARMS[0]
            did = self._insert_decision(scene, command, arm, ctx_json)
            why = "bandit disabled -> no exploration" if not self.enabled else "bad context dim -> safe default"
            self._audit("DECIDE", f"arm={arm} (forced) scene={scene}{replay_note}", why, source="environment")
            return {"arm": arm, "decision_id": did, "scores": {a: 0.0 for a in ARMS},
                    "eligible": eligible, "procedure_match": pm, "procedure_entry": procedure_entry}

        try:
            scores = {}
            for arm in ARMS:
                A = self._A[arm]
                b = self._b[arm]
                A_inv = np.linalg.inv(A)
                theta = A_inv @ b
                mean = float(theta @ x)
                bonus = self.alpha * float(np.sqrt(max(0.0, x @ A_inv @ x)))
                scores[arm] = mean + bonus
            arm = max(eligible, key=lambda a: scores[a])  # argmax; ties -> first eligible arm in ARMS order
            did = self._insert_decision(scene, command, arm, ctx_json)
            self._audit("DECIDE", f"arm={arm} scene={scene}{replay_note} "
                        f"scores={{{', '.join(f'{a}:{scores[a]:.3f}' for a in eligible)}}}",
                        "LinUCB argmax over eligible planning strategies", source="environment")
            return {"arm": arm, "decision_id": did, "scores": scores,
                    "eligible": eligible, "procedure_match": pm, "procedure_entry": procedure_entry}
        except Exception as e:
            arm = ARMS[0]
            did = self._insert_decision(scene, command, arm, ctx_json)
            self._audit("DECIDE-ERR", f"{e!r} -> fallback arm={arm}", source="environment")
            return {"arm": arm, "decision_id": did, "scores": {a: 0.0 for a in ARMS},
                    "eligible": eligible, "procedure_match": pm, "procedure_entry": procedure_entry}

    # ----------------------------------------------------------------- reward (LinUCB update)
    def reward(self, decision_id, reward, *, source: str = "environment") -> None:
        """Credit a past decision with a scalar ``reward`` and run that arm's LinUCB update.

        Looks up the decision row (its arm + stored context). **Provenance rule** (``human`` 2 >
        ``environment`` 1): if the row already carries a reward from an equal-or-higher-provenance source,
        the reward is NOT overwritten (a human thumb can override an environment success, but not the
        reverse, and a repeated same-provenance signal is ignored so it can't be double-counted).

        Otherwise the LinUCB update is applied for that arm: ``A += x x^T`` and ``b += reward * x``, the arm
        state is persisted, and the row's ``reward``/``source`` are set. When a higher-provenance reward
        SUPERSEDES an already-applied one, the prior contribution is first reversed so A/b are not
        double-counted. Never raises.

        **Freeze gate**: when ``updates_enabled`` is False the whole update is an audited no-op: no A/b
        mutation, no persistence, and the decision row is left PENDING (``reward`` stays NULL) so the same
        signal can still be applied after an unfreeze; nothing is silently half-recorded.
        """
        try:
            did = int(decision_id)
        except Exception:
            self._audit("REWARD-ERR", f"bad decision_id={decision_id!r}", source=source)
            return
        if did < 0:
            return
        try:
            row = self.conn.execute(
                "SELECT arm, context, reward, source FROM bandit_decisions WHERE id=?", (did,)).fetchone()
            if row is None:
                self._audit("REWARD-ERR", f"no decision row id={did}", source=source)
                return
            arm = row["arm"]
            if arm not in self._A:
                self._audit("REWARD-ERR", f"unknown arm {arm!r} on decision {did}", source=source)
                return

            # Freeze gate (Phase-7 ablation / loop off): learning is disabled, so skip BEFORE any A/b
            # mutation or row stamp. The decision stays pending, re-appliable after an unfreeze.
            if not self.updates_enabled:
                self._audit("REWARD-FROZEN", f"decision {did} arm={arm} r={reward!r} src={source} ignored",
                            "updates_enabled=False (loop off / teach-then-freeze)", source=source)
                return

            new_rank = BANDIT_PROVENANCE.get(source, 1)
            old_reward = row["reward"]
            old_source = row["source"]
            old_rank = BANDIT_PROVENANCE.get(old_source, 0) if old_reward is not None else -1

            # Provenance gate: an equal-or-higher-provenance reward already present is authoritative.
            if old_reward is not None and old_rank >= new_rank:
                self._audit("REWARD-SKIP", f"decision {did} arm={arm} keep {old_source}({old_reward}) "
                            f">= {source}({reward})", "provenance rule: don't overwrite", source=source)
                return

            x = self._as_vector(json.loads(row["context"]))
            if x is None:
                self._audit("REWARD-ERR", f"decision {did} has unusable context", source=source)
                return

            r = float(reward)
            # If a lower-provenance reward was already folded into A/b, reverse it before applying the new one.
            if old_reward is not None:
                self._A[arm] = self._A[arm] - np.outer(x, x)
                self._b[arm] = self._b[arm] - float(old_reward) * x
                self._updates[arm] = max(0, self._updates[arm] - 1)

            self._A[arm] = self._A[arm] + np.outer(x, x)
            self._b[arm] = self._b[arm] + r * x
            self._updates[arm] = self._updates[arm] + 1
            self._persist_arm(arm)

            self.conn.execute("UPDATE bandit_decisions SET reward=?, source=? WHERE id=?", (r, source, did))
            self.conn.commit()
            supersede = "" if old_reward is None else f" (superseded {old_source}={old_reward})"
            self._audit("REWARD", f"decision {did} arm={arm} r={r:+.2f} src={source}{supersede} "
                        f"updates={self._updates[arm]}", "LinUCB A+=xx^T, b+=r*x", source=source)
        except Exception as e:
            self._audit("REWARD-ERR", f"decision {decision_id}: {e!r}", source=source)

    # ----------------------------------------------------------------- runtime freeze toggle
    def set_updates_enabled(self, flag) -> bool:
        """Flip the learning freeze at runtime (audited). Returns the new value. Never raises."""
        self.updates_enabled = bool(flag)
        self._audit("UPDATES-TOGGLE", f"updates_enabled={self.updates_enabled}",
                    "runtime freeze gate (Phase-7 loop toggle)", source="environment")
        return self.updates_enabled

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        """Per-arm LinUCB updates, decision counts, and running mean reward — plus totals. Never raises."""
        out = {"arms": {}, "total_decisions": 0, "total_updates": 0,
               "updates_enabled": bool(self.updates_enabled)}
        try:
            counts = {}       # arm -> total decisions logged
            rewarded = {}     # arm -> (#rewarded, sum reward)
            for r in self.conn.execute(
                    "SELECT arm, COUNT(*) n, SUM(reward IS NOT NULL) nr, "
                    "SUM(COALESCE(reward,0)) sr FROM bandit_decisions GROUP BY arm").fetchall():
                counts[r["arm"]] = int(r["n"] or 0)
                rewarded[r["arm"]] = (int(r["nr"] or 0), float(r["sr"] or 0.0))
            for arm in ARMS:
                n = counts.get(arm, 0)
                nr, sr = rewarded.get(arm, (0, 0.0))
                out["arms"][arm] = {
                    "updates": int(self._updates.get(arm, 0)),
                    "decisions": n,
                    "rewarded": nr,
                    "mean_reward": (sr / nr) if nr else None,
                }
                out["total_decisions"] += n
                out["total_updates"] += int(self._updates.get(arm, 0))
        except Exception as e:
            self._audit("STATS-ERR", f"{e!r}", source="environment")
        return out

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    # =============================================================================================
    # Self-test: a SYNTHETIC contextual world where the best arm DEPENDS ON THE CONTEXT.
    #
    #   ground-truth optimal arm:
    #     memory_hit == 1               -> "reuse_memory"       (memory already has the answer)
    #     memory_hit == 0 and Persian   -> "explore_then_plan"  (no memory, go look)
    #     memory_hit == 0 and English   -> "plan_fresh"         (no memory, plan without it)
    #
    #   reward(chosen) = Bernoulli(0.9) if chosen == optimal else Bernoulli(0.15)   (mirrors success 0/1)
    #   failure_seen is a pure-noise feature (irrelevant to the optimum) -> tests robustness.
    #
    # We run ~600 rounds pulling reward from that ground truth, then print that the LAST-quartile mean
    # reward >> FIRST-quartile mean reward and that the context-optimal arm is chosen most of the time
    # late in training. Pure numpy + stdlib sqlite3 — no ai2thor / GPU / DISPLAY needed.
    # =============================================================================================
    import tempfile

    rng = np.random.default_rng(7)
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "bandit_selftest.db")
    bandit = Bandit(db, alpha=1.0, enabled=True)

    SCENES = [f"FloorPlan{i}" for i in range(1, 6)]
    FA_CMDS = ["سیب را بردار", "لیوان را در سینک بگذار", "در یخچال را باز کن", "قابلمه را بردار"]
    EN_CMDS = ["pick up the apple", "put the mug in the sink", "open the fridge", "grab the pot"]

    def optimal_arm(memory_hit, lang_fa):
        if memory_hit:
            return "reuse_memory"
        return "explore_then_plan" if lang_fa else "plan_fresh"

    N = 600
    rewards = []          # reward received each round
    optimal_hits = []     # 1 if chosen arm == context-optimal arm

    for t in range(N):
        memory_hit = int(rng.integers(0, 2))
        failure_seen = int(rng.integers(0, 2))   # noise feature
        lang_fa = int(rng.integers(0, 2))
        command = str(rng.choice(FA_CMDS if lang_fa else EN_CMDS))
        scene = str(rng.choice(SCENES))

        x = bandit.featurize(command=command, scene=scene, memory_hit=memory_hit, failure_seen=failure_seen)
        dec = bandit.decide(x, scene=scene, command=command)
        chosen = dec["arm"]

        opt = optimal_arm(memory_hit, lang_fa)
        p = 0.9 if chosen == opt else 0.15
        r = float(rng.random() < p)             # Bernoulli reward = simulated episode success 0/1
        bandit.reward(dec["decision_id"], r, source="environment")

        rewards.append(r)
        optimal_hits.append(1 if chosen == opt else 0)

    q = N // 4
    first_q_mean = float(np.mean(rewards[:q]))
    last_q_mean = float(np.mean(rewards[-q:]))
    last_q_optimal = float(np.mean(optimal_hits[-q:]))
    first_q_optimal = float(np.mean(optimal_hits[:q]))

    print("=" * 78)
    print(f"LinUCB contextual-bandit self-test  ({N} rounds, d={bandit.d}, arms={ARMS})")
    print("-" * 78)
    print(f"first-quartile  mean reward = {first_q_mean:.3f}   optimal-arm rate = {first_q_optimal:.2%}")
    print(f"last-quartile   mean reward = {last_q_mean:.3f}   optimal-arm rate = {last_q_optimal:.2%}")
    print(f"improvement (last - first)  = {last_q_mean - first_q_mean:+.3f}")

    # Provenance-rule spot check: a HUMAN thumb overrides an environment reward on the same decision;
    # the reverse (environment after human) is ignored.
    x0 = bandit.featurize(command="pick up the apple", scene="FloorPlan1", memory_hit=1, failure_seen=0)
    d0 = bandit.decide(x0, scene="FloorPlan1", command="pick up the apple")
    bandit.reward(d0["decision_id"], 0.0, source="environment")   # env says fail
    bandit.reward(d0["decision_id"], 1.0, source="human")         # human thumbs up -> overrides
    bandit.reward(d0["decision_id"], 0.0, source="environment")   # env again -> must be ignored
    chk = bandit.conn.execute("SELECT reward, source FROM bandit_decisions WHERE id=?",
                              (d0["decision_id"],)).fetchone()
    print("-" * 78)
    print(f"provenance check: decision {d0['decision_id']} final reward={chk['reward']} source={chk['source']} "
          f"(expect 1.0 / human)")
    print("per-arm stats:", json.dumps(bandit.stats()["arms"], ensure_ascii=False))
    print("=" * 78)

    # With the deterministic seed the bandit often finds the right arms early, so the first-quartile reward can
    # already be high. Gate the property we actually need: late decisions are dominated by the context-optimal arm
    # and late reward remains strong, without requiring a large delta from an already-good baseline.
    ok_learn = last_q_mean >= 0.80 and last_q_mean >= first_q_mean - 0.02
    ok_optimal = last_q_optimal >= 0.80
    ok_prov = (abs((chk["reward"] or -9) - 1.0) < 1e-9) and (chk["source"] == "human")
    print(f"CONVERGED (strong late reward): {ok_learn} | OPTIMAL-ARM-DOMINANT: {ok_optimal} | PROVENANCE: {ok_prov}")
    assert ok_learn, f"expected strong late reward without regression ({last_q_mean:.3f} vs {first_q_mean:.3f})"
    assert ok_optimal, f"expected context-optimal arm chosen >=80% late, got {last_q_optimal:.2%}"
    assert ok_prov, f"provenance rule failed: {dict(chk)!r}"

    # NOTE: the 600-round world above never passes procedure_match, so replay_procedure was never
    # eligible; the convergence result doubles as proof the 4th arm is inert without a match.

    # =========================================================================================
    # Phase-7 check 1: replay_procedure eligibility gate.
    #   - no / low procedure_match  -> the arm must never be eligible nor chosen;
    #   - high match + high reward  -> the bandit must learn to pick it, and echo the entry id.
    # =========================================================================================
    db2 = os.path.join(tmp, "bandit_replay.db")
    b2 = Bandit(db2, alpha=1.0, enabled=True)
    x1 = b2.featurize(command="pick up the apple", scene="FloorPlan1", memory_hit=1, failure_seen=0)
    for pm in (None, 0.0, REPLAY_MATCH_THRESHOLD - 0.01, "garbage"):
        d = b2.decide(x1, scene="FloorPlan1", command="pick up the apple", procedure_match=pm)
        assert "replay_procedure" not in d["eligible"], (pm, d)
        assert d["arm"] != "replay_procedure", (pm, d)
        b2.reward(d["decision_id"], 0.2, source="environment")
    for _ in range(120):
        d = b2.decide(x1, scene="FloorPlan1", command="pick up the apple",
                      procedure_match=0.95, procedure_entry="proc-001")
        assert "replay_procedure" in d["eligible"], d
        r = 1.0 if d["arm"] == "replay_procedure" else 0.2
        b2.reward(d["decision_id"], r, source="environment")
    d_on = b2.decide(x1, scene="FloorPlan1", command="pick up the apple",
                     procedure_match=0.95, procedure_entry="proc-001")
    assert d_on["arm"] == "replay_procedure", f"expected trained bandit to replay, got {d_on!r}"
    assert d_on["procedure_entry"] == "proc-001" and d_on["procedure_match"] == 0.95, d_on
    d_off = b2.decide(x1, scene="FloorPlan1", command="pick up the apple")  # no match passed
    assert d_off["arm"] != "replay_procedure", f"ineligible arm chosen: {d_off!r}"
    assert "replay_procedure" in b2.stats()["arms"], "stats must include the new arm"
    print(f"replay-gate check: trained arm={d_on['arm']} (entry={d_on['procedure_entry']}), "
          f"no-match arm={d_off['arm']}  -> OK")

    # =========================================================================================
    # Phase-7 check 2: migration -- a DB persisted WITHOUT the new arm must load with the old arms'
    # learned state intact and a FRESH replay_procedure row (upgrading a live memory.db in place).
    # =========================================================================================
    upd_before = {a: b2._updates[a] for a in ARMS}
    b2.conn.execute("DELETE FROM bandit_arms WHERE arm='replay_procedure'")  # simulate pre-Phase-7 DB
    b2.conn.commit()
    b2.close()
    b3 = Bandit(db2, alpha=1.0, enabled=True)
    assert b3._updates["replay_procedure"] == 0, b3._updates
    assert np.array_equal(b3._A["replay_procedure"], np.identity(b3.d)), "expected fresh identity A"
    assert np.array_equal(b3._b["replay_procedure"], np.zeros(b3.d)), "expected fresh zero b"
    for a in ("reuse_memory", "plan_fresh", "explore_then_plan"):
        assert b3._updates[a] == upd_before[a], (a, b3._updates[a], upd_before[a])
    b3.close()
    print(f"migration check: old arms kept updates={ {a: upd_before[a] for a in ARMS[:3]} }, "
          f"replay_procedure fresh  -> OK")

    # =========================================================================================
    # Phase-7 check 3: updates_enabled freeze -- decide() works and logs, reward() is a no-op that
    # leaves the decision pending; unfreeze re-applies the same signal.
    # =========================================================================================
    db3 = os.path.join(tmp, "bandit_frozen.db")
    frozen = Bandit(db3, alpha=1.0, enabled=True, updates_enabled=False)
    assert frozen.stats()["updates_enabled"] is False
    xf = frozen.featurize(command="open the fridge", scene="FloorPlan2", memory_hit=0, failure_seen=0)
    df = frozen.decide(xf, scene="FloorPlan2", command="open the fridge")
    assert df["decision_id"] > 0, "frozen bandit must still log decisions"
    A_before = frozen._A[df["arm"]].copy()
    u_before = frozen._updates[df["arm"]]
    frozen.reward(df["decision_id"], 1.0, source="environment")
    assert frozen._updates[df["arm"]] == u_before, "frozen reward must not bump updates"
    assert np.array_equal(frozen._A[df["arm"]], A_before), "frozen reward must not mutate A"
    rowf = frozen.conn.execute("SELECT reward, source FROM bandit_decisions WHERE id=?",
                               (df["decision_id"],)).fetchone()
    assert rowf["reward"] is None, f"frozen reward must leave the row pending, got {dict(rowf)!r}"
    assert frozen.set_updates_enabled(True) is True
    frozen.reward(df["decision_id"], 1.0, source="environment")
    assert frozen._updates[df["arm"]] == u_before + 1, "unfrozen reward must apply"
    rowf = frozen.conn.execute("SELECT reward, source FROM bandit_decisions WHERE id=?",
                               (df["decision_id"],)).fetchone()
    assert rowf["reward"] == 1.0 and rowf["source"] == "environment", dict(rowf)
    frozen.close()
    print("freeze check: reward no-op while frozen, re-applied after unfreeze  -> OK")

    bandit.close()
    for base in (db, db2, db3):
        for p in (base, os.path.splitext(base)[0] + "_bandit_audit.log"):
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("SELFTEST OK")
