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
