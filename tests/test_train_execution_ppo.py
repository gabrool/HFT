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


def test_adverse_runtime_config_inherits_post_only_gap_from_ppo_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--tape-root", "/tmp/tape",
        "--adverse-signals-npz", "/tmp/adverse.npz",
        "--post-only-gap-ticks", "2",
    ])
    config = _config_from_args(args)
    env_config = _build_env_config(config)

    assert env_config.adverse_runtime_config is not None
    assert env_config.quote_geometry_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.executable_edge.maker_fee_bps == env_config.fill_simulator_config.maker_fee_bps
