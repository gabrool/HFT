from mmrt.cli.train_execution_ppo import build_arg_parser, _build_env_config, _config_from_args, _summary_config


def test_parser_can_disable_l2_trade_dedupe():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--no-dedupe-l2-decrease-with-trade-prints"])
    config = _config_from_args(args)
    env_config = _build_env_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert env_config.fill_simulator_config.queue_model.dedupe_l2_decrease_with_trade_prints is False
    assert _summary_config(config)["dedupe_l2_decrease_with_trade_prints"] is False


def test_parser_dedupe_l2_trade_default_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape"])
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True
