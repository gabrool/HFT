import copy
import pytest

from mmrt.metadata.binance_exchange_info import parse_binance_usdm_exchange_info_symbol


def _payload():
    return {"symbols": [{
        "symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "marginAsset": "USDT",
        "contractType": "PERPETUAL", "status": "TRADING", "pricePrecision": 999, "quantityPrecision": 999,
        "orderTypes": ["LIMIT", "MARKET"], "timeInForce": ["GTC", "GTX"],
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0.10", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }]}


def test_parse_uses_filters_not_precision_fields():
    rules = parse_binance_usdm_exchange_info_symbol(_payload(), symbol="BTCUSDT")
    assert str(rules.tick_size) == "0.10"
    assert str(rules.step_size) == "0.001"
    assert str(rules.min_notional) == "5"
    assert rules.allowed_order_types == ("LIMIT", "MARKET")
    assert "GTX" in rules.allowed_time_in_force


@pytest.mark.parametrize("filter_type", ["PRICE_FILTER", "LOT_SIZE", "MIN_NOTIONAL"])
def test_missing_required_filters_raise(filter_type):
    payload = _payload()
    payload["symbols"][0]["filters"] = [f for f in payload["symbols"][0]["filters"] if f["filterType"] != filter_type]
    with pytest.raises(ValueError, match="missing required Binance filters"):
        parse_binance_usdm_exchange_info_symbol(payload, symbol="BTCUSDT")


def test_requires_symbol_limit_gtx_and_perpetual():
    with pytest.raises(ValueError, match="not found"):
        parse_binance_usdm_exchange_info_symbol(_payload(), symbol="ETHUSDT")
    payload = _payload(); payload["symbols"][0]["orderTypes"] = ["MARKET"]
    with pytest.raises(ValueError, match="LIMIT"):
        parse_binance_usdm_exchange_info_symbol(payload, symbol="BTCUSDT")
    payload = _payload(); payload["symbols"][0]["timeInForce"] = ["GTC"]
    with pytest.raises(ValueError, match="GTX"):
        parse_binance_usdm_exchange_info_symbol(payload, symbol="BTCUSDT")
    payload = _payload(); payload["symbols"][0]["contractType"] = "CURRENT_QUARTER"
    with pytest.raises(ValueError, match="PERPETUAL"):
        parse_binance_usdm_exchange_info_symbol(payload, symbol="BTCUSDT")
