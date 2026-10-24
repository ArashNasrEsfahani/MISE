"""Price table (prices.json) and dollar arithmetic. An unknown Anthropic model raises -- it is never priced at 0."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class PriceTable:
    def __init__(self, table: dict):
        self.table = table

    @classmethod
    def load(cls, path: str | None = None) -> "PriceTable":
        with open(path or os.path.join(HERE, "prices.json"), encoding="utf-8") as f:
            return cls(json.load(f))

    def info(self, provider: str, model: str) -> dict:
        sect = self.table.get(provider) or {}
        if model in sect:
            return sect[model]
        if "_default" in sect:
            return sect["_default"]
        raise KeyError(f"no price for {provider}/{model} in prices.json (add it with a source and date)")

    def cost(self, provider: str, model: str, tokens_in: int, tokens_out: int, actual: bool = False) -> float:
        """USD for ``tokens_in`` input + ``tokens_out`` output tokens (thinking tokens are inside tokens_out)."""
        p = self.info(provider, model)
        return (tokens_in * p["input"] + tokens_out * p["output"]) / 1e6
