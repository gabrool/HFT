from mmrt.cli.execution_defaults import (
    DEFAULT_ADVERSE_HORIZON_US,
    DEFAULT_DECISION_COMPUTE_LATENCY_US,
    DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS,
    DEFAULT_FILL_HORIZON_US,
    DEFAULT_L2_DECREASE_WEIGHT,
    DEFAULT_ORDER_ENTRY_LATENCY_US,
    DEFAULT_ORDER_QTY,
    DEFAULT_POST_ONLY_GAP_TICKS,
    DEFAULT_QTY_EPSILON,
    DEFAULT_QUEUE_MODE,
    DEFAULT_TRADE_AT_LEVEL_WEIGHT,
    DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY,
)
from mmrt.cli.train_adverse_selection import build_arg_parser, _build_adverse_selection_config, _config_from_args, _summary_config
from mmrt.execution.adverse_selection import adverse_label_config_from_config
from mmrt.execution.contracts import QueueModelMode


def test_parser_can_disable_l2_trade_dedupe():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid", "--no-dedupe-l2-decrease-with-trade-prints"])
    config = _config_from_args(args)
    adverse_config = _build_adverse_selection_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert adverse_config.quote.queue_model.dedupe_l2_decrease_with_trade_prints is False
    assert _summary_config(config)["dedupe_l2_decrease_with_trade_prints"] is False


def test_parser_dedupe_l2_trade_default_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid"])
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True


def test_parser_defaults_resolve_to_colocated_balanced_adverse_labels():
    parser = build_arg_parser()
    config = _config_from_args(parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid"]))
    adverse_config = _build_adverse_selection_config(config)
    queue = adverse_config.quote.queue_model
    label_config = adverse_label_config_from_config(adverse_config)
    summary = _summary_config(config)

    assert config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert config.order_qty == DEFAULT_ORDER_QTY
    assert config.fill_horizon_us == DEFAULT_FILL_HORIZON_US
    assert config.adverse_horizon_us == DEFAULT_ADVERSE_HORIZON_US
    assert config.queue_mode == DEFAULT_QUEUE_MODE
    assert config.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert config.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert config.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert config.dedupe_l2_decrease_with_trade_prints is DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS
    assert queue.mode == QueueModelMode.BALANCED
    assert queue.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert queue.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert queue.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert queue.qty_epsilon == DEFAULT_QTY_EPSILON
    assert adverse_config.quote.latency_config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert adverse_config.quote.latency_config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert adverse_config.quote.order_qty == DEFAULT_ORDER_QTY
    assert adverse_config.quote.fill_horizon_us == DEFAULT_FILL_HORIZON_US
    assert adverse_config.quote.adverse_horizon_us == DEFAULT_ADVERSE_HORIZON_US
    assert label_config["queue_mode"] == "balanced"
    assert label_config["l2_decrease_weight"] == DEFAULT_L2_DECREASE_WEIGHT
    assert label_config["trade_at_level_weight"] == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert label_config["unknown_level_queue_ahead_qty"] == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert label_config["qty_epsilon"] == DEFAULT_QTY_EPSILON
    assert label_config["order_entry_latency_us"] == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert label_config["decision_compute_latency_us"] == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert label_config["post_only_gap_ticks"] == DEFAULT_POST_ONLY_GAP_TICKS
    assert label_config["order_qty"] == DEFAULT_ORDER_QTY
    assert label_config["fill_horizon_us"] == DEFAULT_FILL_HORIZON_US
    assert label_config["adverse_horizon_us"] == DEFAULT_ADVERSE_HORIZON_US
    assert summary["queue_mode"] == "balanced"
    assert summary["order_entry_latency_us"] == DEFAULT_ORDER_ENTRY_LATENCY_US
