from pathlib import Path


PRODUCTION_ROOT = Path("mmrt")

FORBIDDEN_SNIPPETS = (
    "tick_size: float = 0.1",
    "step_size: float = 0.001",
    "min_notional: float = 5.0",
    "parser.add_argument(\"--tick-size\"",
    "parser.add_argument(\"--step-size\"",
    "parser.add_argument(\"--min-notional\"",
    "parser.add_argument('--tick-size'",
    "parser.add_argument('--step-size'",
    "parser.add_argument('--min-notional'",
)


def test_production_code_has_no_hardcoded_execution_symbol_rule_defaults():
    for path in PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            assert snippet not in text, f"{snippet!r} found in {path}"


def test_runtime_execution_modules_do_not_parse_exchange_info():
    paths = [
        Path("mmrt/execution/quote_geometry.py"),
        Path("mmrt/execution/fill_sim.py"),
        Path("mmrt/execution/env.py"),
        *Path("mmrt/rl").rglob("*.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "exchangeInfo" not in text, f"exchangeInfo reference found in {path}"
        assert "binance_exchange_info" not in text, f"binance_exchange_info reference found in {path}"
