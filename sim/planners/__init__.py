"""sim/planners -- one PlannerProvider interface over Gemma (Ollama), Anthropic, OpenRouter and the Codex CLI
(final test, stage P1). Offline-testable; keys only from the environment / an env file at call time.

    from planners import make_planner, PlannerContract, PlannerRunner, CostLedger, PlannerSkip
    p = make_planner("opus", ledger=CostLedger.smoke())      # raises PlannerSkip when ANTHROPIC_API_KEY is unset
"""
from .base import (CostCeilingExceeded, PlannerError, PlannerProvider, PlannerSkip, TransportFailure,  # noqa: F401
                   sha256_json)
from .cache import ADAPTER_VERSION, ResponseCache                                                      # noqa: F401
from .contract import (PLAN_SCHEMA, PROMPT_VERSION, PlannerContract, flat_prompt, parse_reply,        # noqa: F401
                       prompt_sha256, schema_problems)
from .ledger import SMOKE_CEILINGS, CostLedger                                                         # noqa: F401
from .prices import PriceTable                                                                         # noqa: F401
from .runner import PlannerRunner                                                                      # noqa: F401

#: arm key -> (provider family, model id). OpenRouter / Codex models are chosen by the caller: "openrouter:<id>",
#: "codex:<gpt-6-sol|gpt-6-astra|gpt-6-luna|gpt-5.6-terra>".
ARMS = {"gemma": ("ollama", "gemma4:e2b-it-q4_K_M"), "opus": ("anthropic", "claude-opus-5-5"),
        "sonnet": ("anthropic", "claude-sonnet-5-5"), "haiku": ("anthropic", "claude-haiku-4-5-20251001")}


def make_planner(spec: str, **kw) -> PlannerProvider:
    """Build a provider from an arm key ('gemma', 'opus', 'sonnet', 'haiku', 'openrouter:<model>', 'codex:<model>').
    Construction never needs a key; the first live call raises PlannerSkip when one is missing."""
    if ":" in spec and spec.split(":", 1)[0] in ("openrouter", "codex"):
        fam, model = spec.split(":", 1)
    elif spec in ARMS:
        fam, model = ARMS[spec]
    else:
        raise ValueError(f"unknown planner {spec!r}; use {', '.join(ARMS)}, openrouter:<model> or codex:<model>")
    if fam == "ollama":
        from .ollama_provider import OllamaPlanner
        kw.setdefault("strict_schema", True)        # prereg A7: one canonical output contract for every arm
        return OllamaPlanner(model, **kw)
    if fam == "anthropic":
        from .anthropic_provider import AnthropicPlanner
        return AnthropicPlanner(model, **kw)
    if fam == "openrouter":
        from .openrouter_provider import OpenRouterPlanner
        return OpenRouterPlanner(model, **kw)
    from .codex_provider import CodexCliPlanner
    return CodexCliPlanner(model, **kw)
