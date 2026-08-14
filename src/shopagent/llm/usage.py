"""Token usage and cost tracking per session (D1).

Deliberately free of any OpenAI SDK dependency — this module takes plain
numbers from the outside, so it is testable without a single network call.
Whoever calls the LLM extracts `usage` from the response and passes it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Prices in USD per 1M tokens: model_id -> (input, output, cached_input).
# Source: https://platform.openai.com/pricing + pricepertoken.com
# Last checked: 2026-08-14
#
# Cached input is a separate price, NOT a percentage of the full one — the
# discount differs per model (the gpt-5.6-* family pays 10% of full input
# price, gpt-4o-mini pays 50%). A global constant would silently miscalculate.
#
# A missing model here is not an error — its cost is 0.0 and it lands in
# UsageTracker.unknown_models, so an understated cost never passes silently.
PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-sol": (5.00, 30.00, 0.50),
    "gpt-5.6-terra": (2.00, 12.00, 0.20),
    "gpt-5.6-luna": (0.20, 1.20, 0.02),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
}


@dataclass
class CallUsage:
    """Usage of a single LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    # Subset of prompt_tokens that hit the cache. NOT added to the total —
    # these tokens are already counted in prompt_tokens, they are merely
    # billed at a lower rate.
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        """Cost of this call; 0.0 if the model is not in PRICING.

        An unknown model does not raise — one unrecognised model must never
        bring down a conversation. UsageTracker makes sure that zero does not
        go unnoticed.
        """
        pricing = PRICING.get(self.model)
        if pricing is None:
            return 0.0
        input_per_1m, output_per_1m, cached_input_per_1m = pricing
        # cached_tokens is meant to be a subset of prompt_tokens. Bad upstream
        # data or a hand-written call can break that, and an unclamped value
        # yields a negative cost that quietly lowers the session total. The
        # field itself is left untouched so the anomaly stays visible.
        cached_tokens = min(max(self.cached_tokens, 0), max(self.prompt_tokens, 0))
        uncached_tokens = self.prompt_tokens - cached_tokens
        return (
            uncached_tokens * input_per_1m
            + cached_tokens * cached_input_per_1m
            + self.completion_tokens * output_per_1m
        ) / 1_000_000


@dataclass
class UsageTracker:
    """Accumulates usage across one session."""

    calls: list[CallUsage] = field(default_factory=list)
    unknown_models: set[str] = field(default_factory=set)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> CallUsage:
        """Record one call and return its CallUsage."""
        call = CallUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        if model not in PRICING:
            self.unknown_models.add(model)
        self.calls.append(call)
        return call

    @property
    def total_tokens(self) -> int:
        return sum(call.total_tokens for call in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    def summary(self) -> str:
        """One-line session summary, warning if the cost is incomplete."""
        line = (
            f"{len(self.calls)} calls · {self.total_tokens:,} tokens · "
            f"${self.total_cost_usd:.6f}"
        )
        if self.unknown_models:
            names = ", ".join(sorted(self.unknown_models))
            line += (
                f" · WARNING: no pricing for {names}"
                f" — actual cost is higher than shown"
            )
        return line
