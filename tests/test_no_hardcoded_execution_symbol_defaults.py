from pathlib import Path


def test_execution_tape_builder_has_no_hardcoded_symbol_rule_defaults():
    forbidden = (
        "tick_size: float = 0.1",
        "step_size: float = 0.001",
        "min_notional: float = 5.0",
        "--tick-size",
        "--step-size",
        "--min-notional",
    )
    roots = [Path("mmrt/cli/build_execution_tape.py"), Path("mmrt/execution/execution_tape.py")]
    for path in roots:
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text
