"""Tests for shopagent.llm.usage.

Prices are injected via monkeypatch — tests must not depend on real model
prices, since those change and would make the suite fail for reasons that have
nothing to do with the code.
"""

import pytest

from shopagent.llm import usage as usage_mod
from shopagent.llm.usage import CallUsage, UsageTracker

FAKE_MODEL = "fake-model"
UNKNOWN_MODEL = "nonexistent-model"


@pytest.fixture
def fake_pricing(monkeypatch):
    """FAKE_MODEL: $1.00 input, $2.00 output, $0.10 cached input (= 10%)."""
    monkeypatch.setitem(usage_mod.PRICING, FAKE_MODEL, (1.0, 2.0, 0.10))
    return FAKE_MODEL


def test_cost_usd_for_known_model(fake_pricing):
    call = CallUsage(
        model=FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500
    )
    # (1000 * 1.0 + 500 * 2.0) / 1_000_000
    assert call.cost_usd == pytest.approx(0.002)
    assert call.total_tokens == 1_500


def test_cost_usd_for_unknown_model_is_zero_and_does_not_raise():
    call = CallUsage(
        model=UNKNOWN_MODEL, prompt_tokens=100, completion_tokens=50
    )
    assert call.cost_usd == 0.0


def test_unknown_model_lands_in_unknown_models():
    tracker = UsageTracker()
    tracker.record(UNKNOWN_MODEL, 100, 50)

    assert tracker.unknown_models == {UNKNOWN_MODEL}
    assert tracker.total_cost_usd == 0.0
    assert tracker.total_tokens == 150


def test_known_model_does_not_land_in_unknown_models(fake_pricing):
    tracker = UsageTracker()
    tracker.record(FAKE_MODEL, 100, 50)

    assert tracker.unknown_models == set()


def test_record_returns_call_usage(fake_pricing):
    tracker = UsageTracker()
    call = tracker.record(FAKE_MODEL, 1_000, 500)

    assert isinstance(call, CallUsage)
    assert call.model == FAKE_MODEL
    assert call.cost_usd == pytest.approx(0.002)
    assert tracker.calls == [call]


def test_totals_across_multiple_calls(fake_pricing):
    tracker = UsageTracker()
    tracker.record(FAKE_MODEL, 1_000, 500)      # 1500 tokens, $0.002
    tracker.record(FAKE_MODEL, 2_000, 1_000)    # 3000 tokens, $0.004
    tracker.record(UNKNOWN_MODEL, 100, 100)     #  200 tokens, $0.000

    assert tracker.total_tokens == 4_700
    assert tracker.total_cost_usd == pytest.approx(0.006)
    assert len(tracker.calls) == 3


def test_summary_format_without_unknown_models(fake_pricing):
    tracker = UsageTracker()
    tracker.record(FAKE_MODEL, 1_000, 500)

    summary = tracker.summary()

    assert summary == "1 calls · 1,500 tokens · $0.002000"
    assert "WARNING" not in summary


def test_summary_warns_about_unknown_model(fake_pricing):
    tracker = UsageTracker()
    tracker.record(FAKE_MODEL, 1_000, 500)
    tracker.record(UNKNOWN_MODEL, 100, 50)

    summary = tracker.summary()

    assert "WARNING" in summary
    assert UNKNOWN_MODEL in summary
    # the cost part must still be there
    assert "$0.002000" in summary


def test_summary_lists_every_unknown_model():
    tracker = UsageTracker()
    tracker.record("model-b", 10, 10)
    tracker.record("model-a", 10, 10)

    summary = tracker.summary()

    assert "model-a" in summary
    assert "model-b" in summary


def test_empty_tracker():
    tracker = UsageTracker()

    assert tracker.total_tokens == 0
    assert tracker.total_cost_usd == 0.0
    assert tracker.summary() == "0 calls · 0 tokens · $0.000000"


def test_fixture_does_not_pollute_pricing():
    """Sanity: the fixture must not leave FAKE_MODEL in PRICING."""
    assert FAKE_MODEL not in usage_mod.PRICING


# --- cached input tokens -------------------------------------------------


def test_cached_tokens_default_to_zero(fake_pricing):
    """With no cache the result must match the pre-cache behaviour exactly."""
    without_field = CallUsage(FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500)
    explicit_zero = CallUsage(
        FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500, cached_tokens=0
    )

    assert without_field.cached_tokens == 0
    assert without_field.cost_usd == pytest.approx(0.002)
    assert explicit_zero.cost_usd == without_field.cost_usd


def test_cached_tokens_lower_the_cost(fake_pricing):
    without_cache = CallUsage(FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500)
    with_cache = CallUsage(
        FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500, cached_tokens=800
    )

    assert with_cache.cost_usd < without_cache.cost_usd
    # 200 * 1.0 + 800 * 0.10 + 500 * 2.0 = 200 + 80 + 1000 = 1280
    assert with_cache.cost_usd == pytest.approx(0.00128)


def test_full_cache_is_billed_at_the_cached_rate(fake_pricing):
    """cached_tokens == prompt_tokens -> the input part uses the cached price.

    For FAKE_MODEL the cached input is 10% of full ($0.10 vs $1.00).
    """
    full_price = CallUsage(FAKE_MODEL, prompt_tokens=1_000, completion_tokens=0)
    fully_cached = CallUsage(
        FAKE_MODEL, prompt_tokens=1_000, completion_tokens=0, cached_tokens=1_000
    )

    assert fully_cached.cost_usd == pytest.approx(full_price.cost_usd * 0.10)
    # 1000 * 0.10 / 1M
    assert fully_cached.cost_usd == pytest.approx(0.0001)


def test_cached_tokens_are_not_counted_in_total_tokens(fake_pricing):
    call = CallUsage(
        FAKE_MODEL, prompt_tokens=1_000, completion_tokens=500, cached_tokens=800
    )

    assert call.total_tokens == 1_500


def test_record_accepts_cached_tokens(fake_pricing):
    tracker = UsageTracker()
    call = tracker.record(FAKE_MODEL, 1_000, 500, cached_tokens=800)

    assert call.cached_tokens == 800
    assert tracker.total_cost_usd == pytest.approx(0.00128)
    assert tracker.total_tokens == 1_500


def test_cached_tokens_on_unknown_model_still_cost_zero():
    call = CallUsage(
        UNKNOWN_MODEL, prompt_tokens=1_000, completion_tokens=500, cached_tokens=800
    )

    assert call.cost_usd == 0.0


def test_cached_discount_is_per_model(monkeypatch):
    """Two models with the same input/output but different cached rates.

    This is why the cached price lives in PRICING instead of being a global
    constant: identical usage must produce different costs.
    """
    # same input ($1.00) and output ($2.00), cached 10% vs 50%
    monkeypatch.setitem(usage_mod.PRICING, "cheap-cache", (1.0, 2.0, 0.10))
    monkeypatch.setitem(usage_mod.PRICING, "pricey-cache", (1.0, 2.0, 0.50))

    usage_kwargs = dict(prompt_tokens=1_000, completion_tokens=0, cached_tokens=1_000)
    cheap = CallUsage("cheap-cache", **usage_kwargs)
    pricey = CallUsage("pricey-cache", **usage_kwargs)

    assert cheap.cost_usd == pytest.approx(0.0001)    # 1000 * 0.10 / 1M
    assert pricey.cost_usd == pytest.approx(0.0005)   # 1000 * 0.50 / 1M
    assert pricey.cost_usd == pytest.approx(cheap.cost_usd * 5)


def test_real_prices_have_three_entries():
    """Sanity: every PRICING entry must be (input, output, cached_input)."""
    for model, pricing in usage_mod.PRICING.items():
        assert len(pricing) == 3, f"{model} has no cached input price"
        assert pricing[2] < pricing[0], f"{model}: cached is not cheaper than full"
