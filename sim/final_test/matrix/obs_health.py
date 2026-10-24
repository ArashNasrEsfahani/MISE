"""A34-A2 (signed 2026-10-15) rule 2(b): observation-pipeline HEALTH and EFFECTIVE SETTINGS evidence for one episode (Astra ACC2-IMPL-r1 3).

Pure instrumentation, behaviour-preserving: `install()` wraps, at class level and once per process, `perception.TwoStageDetector.detect`
(counts calls and records the effective detector configuration it ran with) and `perception.StateEstimator.annotate` (counts calls and
exceptions; an exception is RE-RAISED unchanged, so the caller's own handling -- TwoStageDetector swallows a state failure -- is exactly
as before). Nothing else is touched; no return value is altered. `reset()` before an episode, `snapshot()` right after the start sweep.

The capture (Phase A) and the execution (Phase C) both record {"schema", "treatment_id", "sweep_prims", "health": snapshot()}; reset_gate's
`pipeline_problems` / `pipeline_settings` judge it. Missing evidence never passes an override.
"""
from __future__ import annotations

import os
import threading

SCHEMA = "obs-pipeline/1"
_LOCK = threading.Lock()
_S = {"detect_calls": 0, "state_calls": 0, "state_errors": 0, "configs": set()}
_INSTALLED = {"ok": False}


def _base(p):
    return os.path.basename(str(p)) if p else None


def install() -> bool:
    """Idempotent. Returns True when the wrappers are in place (False if perception cannot be imported: the evidence is then absent)."""
    if _INSTALLED["ok"]:
        return True
    try:
        import perception as P
    except Exception:                                          # noqa: BLE001 -- no perception: no evidence (an override stays blocked)
        return False
    det0, ann0 = P.TwoStageDetector.detect, P.StateEstimator.annotate
    if getattr(det0, "_obs_health", False):
        _INSTALLED["ok"] = True
        return True

    def detect(self, frame_rgb, conf: float = 0.25, inspecting=None):
        cfg = (str(getattr(self, "mode", None)), _base(getattr(getattr(self, "primary", None), "weights_path", None)),
               _base(getattr(getattr(self, "specialist", None), "weights_path", None)), getattr(self, "states", None) is not None,
               float(conf))
        with _LOCK:
            _S["detect_calls"] += 1
            _S["configs"].add(cfg)
        return det0(self, frame_rgb, conf=conf, inspecting=inspecting)

    def annotate(self, frame_rgb, dets):
        with _LOCK:
            _S["state_calls"] += 1
        try:
            return ann0(self, frame_rgb, dets)
        except Exception:
            with _LOCK:
                _S["state_errors"] += 1
            raise

    detect._obs_health = True
    annotate._obs_health = True
    P.TwoStageDetector.detect = detect
    P.StateEstimator.annotate = annotate
    _INSTALLED["ok"] = True
    return True


def reset() -> None:
    with _LOCK:
        _S.update(detect_calls=0, state_calls=0, state_errors=0, configs=set())


def snapshot() -> dict | None:
    """The health counters since the last reset(); None when the instrumentation is not installed (= no evidence)."""
    if not _INSTALLED["ok"]:
        return None
    with _LOCK:
        return {"detect_calls": _S["detect_calls"], "state_calls": _S["state_calls"], "state_errors": _S["state_errors"],
                "configs": sorted([list(c) for c in _S["configs"]])}


def record(treatment_id, sweep_prims) -> dict:
    """The per-episode observation-pipeline record (both phases build it the same way)."""
    return {"schema": SCHEMA, "treatment_id": treatment_id, "sweep_prims": sweep_prims, "health": snapshot()}
