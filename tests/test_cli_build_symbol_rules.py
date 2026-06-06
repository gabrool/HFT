import json

import pytest

from mmrt.cli.build_symbol_rules import build_arg_parser, main


def _exchange_info(path):
    payload = {"symbols": [{
        "symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "marginAsset": "USDT",
        "contractType": "PERPETUAL", "status": "TRADING", "orderTypes": ["LIMIT"], "timeInForce": ["GTX"],
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0.10", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }]}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_build_symbol_rules_writes_artifact_refuses_overwrite_and_prints_json(tmp_path, capsys):
    source = _exchange_info(tmp_path / "exchangeInfo.json")
    out = tmp_path / "BTCUSDT.symbol_rules.json"
    rc = main(["--symbol", "BTCUSDT", "--exchange-info-json", str(source), "--output-json", str(out)])
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert printed["status"] == "ok"
    assert saved["tick_size"] == "0.10"
    with pytest.raises(FileExistsError):
        main(["--symbol", "BTCUSDT", "--exchange-info-json", str(source), "--output-json", str(out)])


def test_parser_has_no_network_option():
    help_text = build_arg_parser().format_help()
    assert "http" not in help_text.lower()
    assert "url" not in help_text.lower()
