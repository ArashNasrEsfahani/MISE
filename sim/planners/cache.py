"""Content-addressed response cache: planning is replayable.

The key hashes a canonical manifest: provider, model id, the role-labelled prompt hash, the effective decoding, the
replicate index (independent repeats r=0,1,2 have separate entries), the canonical schema hash, the stage
("plan" / "repair"; a repair prompt also contains the failed reply), the adapter + prompt-contract versions and a
frozen namespace (a mutable model alias cannot be detected by a hash, so a new acquisition = a new namespace).
Transport failures are never cached. A cache entry keeps the ORIGINAL cost and latency; a replay costs 0 USD and is
never counted as another observation.
"""
from __future__ import annotations

import json
import os
import threading
import time

from .base import sha256_json

ADAPTER_VERSION = "planners-v1"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), "final_test", "cache")


class ResponseCache:
    def __init__(self, root: str | None = None, namespace: str = "default"):
        self.root, self.namespace = root or DEFAULT_ROOT, namespace

    def key(self, provider, model, prompt_sha, decoding, replicate, schema, stage="plan") -> str:
        from .contract import PROMPT_VERSION
        return sha256_json({"adapter": ADAPTER_VERSION, "prompt_version": PROMPT_VERSION, "namespace": self.namespace,
                            "provider": provider, "model": model, "prompt": prompt_sha, "decoding": decoding,
                            "replicate": replicate, "schema": sha256_json(schema), "stage": stage})

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".json")

    def get(self, key: str):
        try:
            with open(self._path(key), encoding="utf-8") as f:
                d = json.load(f)
            return d["response"] if d.get("key") == key and isinstance(d.get("response"), dict) else None
        except (OSError, ValueError, KeyError):
            return None                                             # absent or corrupt -> a miss

    def put(self, key: str, response: dict, manifest: dict):
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f".{os.getpid()}.{threading.get_ident()}.tmp"      # plan-first: several worker threads of one process write at once
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"key": key, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "adapter": ADAPTER_VERSION,
                       "namespace": self.namespace, "manifest": manifest, "response": response},
                      f, ensure_ascii=False)
        for k in range(6):                                          # Windows refuses a replace while another thread has the target open: retry briefly
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if k == 5:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise
                time.sleep(0.01 * (k + 1))
