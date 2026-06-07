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
    "is_fillable_at(",
    "prev_level_qty",
    "curr_level_qty",
    "request_local_ts_us",
    "order_effective_local_ts_us",
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
    "pending_cancel_request_count",
)


def test_no_stale_fill_sim_trade_timestamp_fill_path_removed():
    text = Path("mmrt/execution/fill_sim.py").read_text(encoding="utf-8")
    assert "local_ts_us=trade.local_ts_us" not in text


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


def test_dedupe_cli_flag_exposed_in_relevant_clis():
    paths = [
        Path("mmrt/cli/audit_execution_sim.py"),
        Path("mmrt/cli/train_execution_ppo.py"),
        Path("mmrt/cli/evaluate_execution_policy.py"),
        Path("mmrt/cli/train_adverse_selection.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "--no-dedupe-l2-decrease-with-trade-prints" in text
        assert "dedupe_l2_decrease_with_trade_prints=not args.no_dedupe_l2_decrease_with_trade_prints" in text


def test_place_orders_from_quote_uses_side_specific_effective_keys():
    source = Path("mmrt/execution/fill_sim.py").read_text(encoding="utf-8")
    place_orders_body = source.split("def place_orders_from_quote(", 1)[1].split("def _new_order", 1)[0]
    assert "bid_effective_key" in place_orders_body
    assert "ask_effective_key" in place_orders_body


def test_same_side_replacement_uses_activation_style_key_after_cancel():
    source = Path("mmrt/execution/fill_sim.py").read_text(encoding="utf-8")
    assert "_activation_key_after_cancel" in source
    assert "MAX_EVENT_SEQ" in source
