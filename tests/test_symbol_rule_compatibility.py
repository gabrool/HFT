from decimal import Decimal
import pytest

from mmrt.metadata.rule_compatibility import RuleCompatibilityAccumulator, RuleCompatibilityConfig, RuleCompatibilityMode
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode


def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures", symbol="BTCUSDT", mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC", quote_asset="USDT", margin_asset="USDT", contract_type="PERPETUAL", status="TRADING",
        tick_size=Decimal("0.1"), min_price=Decimal("0.1"), max_price=Decimal("1000000"),
        step_size=Decimal("0.001"), min_qty=Decimal("0.001"), max_qty=Decimal("100"), min_notional=Decimal("5"),
        allowed_order_types=("LIMIT",), allowed_time_in_force=("GTX",),
    )


def test_warn_mode_reports_but_does_not_raise_and_bounds_examples():
    acc = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(max_examples=1))
    acc.observe_price(100.0, source="p")
    acc.observe_qty(0.001, source="q")
    acc.observe_price(100.05, source="p")
    acc.observe_qty(0.0015, source="q")
    report = acc.report()
    assert report.status == "warning"
    assert report.price_grid_violation_count == 1
    assert report.qty_grid_violation_count == 1
    assert len(report.examples) == 1


def test_noise_within_tolerance_ok_and_off_does_not_track():
    acc = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(price_tolerance_ticks=1e-4))
    acc.observe_price(100.0000000001, source="p")
    assert acc.report().status == "ok"
    off = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(mode=RuleCompatibilityMode.OFF))
    off.observe_price(100.05, source="p")
    assert off.report().price_count == 0


def test_strict_raises_on_violation():
    acc = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(mode=RuleCompatibilityMode.STRICT))
    acc.observe_price(100.05, source="p")
    with pytest.raises(ValueError, match="strict mode"):
        acc.report()


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -1.0, "1"])
def test_compatibility_config_rejects_bad_tolerances(bad):
    with pytest.raises(ValueError):
        RuleCompatibilityConfig(price_tolerance_ticks=bad)
    with pytest.raises(ValueError):
        RuleCompatibilityConfig(qty_tolerance_steps=bad)


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, "10"])
def test_compatibility_config_rejects_bad_max_examples(bad):
    with pytest.raises(ValueError):
        RuleCompatibilityConfig(max_examples=bad)


def _accumulator_state(acc: RuleCompatibilityAccumulator) -> dict:
    return {
        "price_count": acc.price_count,
        "qty_count": acc.qty_count,
        "price_violations": acc.price_violations,
        "qty_violations": acc.qty_violations,
        "max_price_residual": acc.max_price_residual,
        "max_qty_residual": acc.max_qty_residual,
        "min_price": acc.min_price,
        "max_price": acc.max_price,
        "min_qty": acc.min_qty,
        "max_qty": acc.max_qty,
        "examples": acc.examples,
        "report": acc.report().to_dict(),
    }


def test_array_observers_match_per_value_observers_exactly():
    import numpy as np

    prices = np.asarray([100.0, 100.1, 100.05, 100.1, 100.05, 99.9, 100.0000000001, 123456.7], dtype=np.float64)
    qtys = np.asarray([0.001, 0.0015, 0.002, 0.0015, 1.0, 0.001, 0.30000000004, 0.001], dtype=np.float64)
    ts = np.arange(1_000, 1_000 + prices.size, dtype=np.int64)

    scalar = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(max_examples=3))
    for i in range(prices.size):
        scalar.observe_price(float(prices[i]), source="p", local_ts_us=int(ts[i]))
    for i in range(qtys.size):
        scalar.observe_qty(float(qtys[i]), source="q", local_ts_us=int(ts[i]))

    vector = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig(max_examples=3))
    vector.observe_price_array(prices, source="p", local_ts_us=ts)
    vector.observe_qty_array(qtys, source="q", local_ts_us=ts)

    assert _accumulator_state(vector) == _accumulator_state(scalar)


def test_array_observers_match_per_value_observers_across_chunks():
    import numpy as np

    rng = np.random.default_rng(7)
    grid = np.round(rng.integers(1, 2_000_000, 500) * 0.1, 1)
    off_grid = grid[:25] + 0.05
    prices = np.concatenate([grid, off_grid])
    rng.shuffle(prices)
    ts = np.arange(10_000, 10_000 + prices.size, dtype=np.int64)

    scalar = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig())
    for i in range(prices.size):
        scalar.observe_price(float(prices[i]), source="p", local_ts_us=int(ts[i]))

    vector = RuleCompatibilityAccumulator(_rules(), RuleCompatibilityConfig())
    for start in range(0, prices.size, 100):
        vector.observe_price_array(prices[start : start + 100], source="p", local_ts_us=ts[start : start + 100])

    assert _accumulator_state(vector) == _accumulator_state(scalar)
