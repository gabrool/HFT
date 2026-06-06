from pathlib import Path


PRODUCTION_ROOT = Path("mmrt")

FORBIDDEN_PRODUCTION_TOKENS = (
    "policy_log_std_init",
    "policy_log_std_min",
    "policy_log_std_max",
    "action_log_std",
    "action_mean",
    "min_distance_ticks",
    "replace_orders_from_quote",
    "book_event",
    "BOOK_EVENT",
    "event_progress",
    "time_since_trade_us",
    "regime_volume_ewma",
    "trade_impact_half_life_proxy",
    "vwap_vs_mid_bps",
    "spread_z_",
    "depth_5bps_z_",
    "return_std_bps_200000us",
    "max_abs_return_bps_500000us",
    "depth_imbalance_realized_vol_1000000us",
)


def test_no_stale_execution_migration_tokens_in_production():
    offenders = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_PRODUCTION_TOKENS:
            if token in text:
                for line_no, line in enumerate(text.splitlines(), 1):
                    if token in line:
                        offenders.append(f"{path}:{line_no}: {line.strip()}")
    assert offenders == []
