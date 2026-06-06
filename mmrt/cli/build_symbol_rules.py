"""Build normalized local symbol-rules artifacts from Binance exchangeInfo JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mmrt.metadata.binance_exchange_info import load_binance_usdm_exchange_info_symbol_rules
from mmrt.metadata.symbol_rules import SymbolRuleMode, write_symbol_rules_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a local Binance USD-M exchangeInfo JSON symbol into rules JSON.")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange-info-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in SymbolRuleMode], default=SymbolRuleMode.CURRENT_RULES_REPLAY.value)
    parser.add_argument("--captured-at-utc")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rules = load_binance_usdm_exchange_info_symbol_rules(
        Path(args.exchange_info_json),
        symbol=args.symbol,
        exchange=args.exchange,
        mode=SymbolRuleMode(args.mode),
        captured_at_utc=args.captured_at_utc,
    )
    write_symbol_rules_json(args.output_json, rules, overwrite=args.overwrite)
    print(json.dumps({"status": "ok", "symbol_rules": rules.to_dict()}, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
