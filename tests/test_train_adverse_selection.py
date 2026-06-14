from mmrt.cli.train_adverse_selection import build_arg_parser, _build_adverse_selection_config, _config_from_args, _summary_config


def test_parser_can_disable_l2_trade_dedupe():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid-npz", "/tmp/tape/decision_grid.npz", "--no-dedupe-l2-decrease-with-trade-prints"])
    config = _config_from_args(args)
    adverse_config = _build_adverse_selection_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert adverse_config.quote.queue_model.dedupe_l2_decrease_with_trade_prints is False
    assert _summary_config(config)["dedupe_l2_decrease_with_trade_prints"] is False


def test_parser_dedupe_l2_trade_default_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid-npz", "/tmp/tape/decision_grid.npz"])
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True
