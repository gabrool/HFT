from decimal import Decimal
import json

from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode, canonical_symbol_rules_json


def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures",
        symbol="BTCUSDT",
        mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_type="PERPETUAL",
        status="TRADING",
        tick_size=Decimal("0.10"),
        min_price=Decimal("0.10"),
        max_price=Decimal("1000000"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        min_notional=Decimal("5.0"),
        allowed_order_types=("LIMIT",),
        allowed_time_in_force=("GTC", "GTX"),
        source_sha256="abc",
    )


def test_symbol_rules_roundtrip_preserves_decimal_strings_and_spec():
    rules = _rules()
    payload = json.loads(canonical_symbol_rules_json(rules))
    assert payload["tick_size"] == "0.10"
    assert payload["step_size"] == "0.001"
    restored = ExchangeSymbolRules.from_dict(payload)
    assert restored == rules
    spec = restored.to_symbol_spec()
    assert spec.tick_size == 0.1
    assert spec.step_size == 0.001
    assert spec.min_notional == 5.0
    assert restored.source_sha256 == "abc"
