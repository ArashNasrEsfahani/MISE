"""Fake token counter for the package selftests (PACKAGE-SPEC 9.4): two 'tokenizers', no network.

counts(t) = {"gemma": ceil(len/3), "anthropic": ceil(len/2.5)}; Persian text is weighted 1.7. It has the same surface as
final_test.c3.real_counter.RealMaxCounter (counts, __call__, tokenizers, header, header_cost) so the builder can be tested
without Ollama or the API. Selftests only; production passes the real counter.
"""
from __future__ import annotations

import math
import re

_FA = re.compile(r"[؀-ۿ]")


class FakeCounter:
    tokenizers = ["gemma", "anthropic"]

    def __init__(self, header: str = "", gemma_div: float = 3.0, anth_div: float = 2.5):
        self.header, self.gemma_div, self.anth_div = header, gemma_div, anth_div
        self.n_calls = 0

    @property
    def header_cost(self) -> int:
        return max(self.counts(self.header + "\n.").values())

    def counts(self, text) -> dict:
        self.n_calls += 1
        if not text or not str(text).strip():
            return {"gemma": 1, "anthropic": 1} if text else {"gemma": 0, "anthropic": 0}
        t = str(text)
        w = 1.7 if _FA.search(t) else 1.0
        return {"gemma": int(math.ceil(len(t) / self.gemma_div * w)), "anthropic": int(math.ceil(len(t) / self.anth_div * w))}

    def __call__(self, text) -> int:
        if not text:
            return 0
        return max(1, max(self.counts(text).values()))
