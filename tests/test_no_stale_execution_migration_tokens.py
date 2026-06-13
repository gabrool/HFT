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

def test_no_stale_adverse_selection_training_constants():
    source = Path("mmrt/cli/train_adverse_selection.py").read_text(encoding="utf-8")
    assert "_BINARY_TARGETS" not in source


def test_no_stale_adverse_selection_npz_writer():
    source = Path("mmrt/cli/train_adverse_selection.py").read_text(encoding="utf-8")
    assert "_write_npz_atomic" not in source
    assert "np.savez(f" not in source


def test_no_legacy_adverse_selection_quote_distance_paths():
    production_paths = [
        Path("mmrt/execution/adverse_selection.py"),
        Path("mmrt/cli/train_adverse_selection.py"),
    ]
    offenders = []
    for path in production_paths:
        text = path.read_text(encoding="utf-8")
        for token in ("quote_distance_ticks", "--quote-distance-ticks"):
            if token in text:
                offenders.append(f"{path}: {token}")
    assert offenders == []


def test_executable_edge_uses_signed_spread_capture():
    source = Path("mmrt/execution/executable_edge.py").read_text(encoding="utf-8")
    assert "max(mid_tick - price_tick, 0.0)" not in source
    assert "max(price_tick - mid_tick, 0.0)" not in source


def test_adverse_selection_signal_build_cli_has_no_rl_dependencies():
    source = Path("mmrt/cli/build_adverse_selection_signals.py").read_text(encoding="utf-8")
    for forbidden in ("mmrt.rl", "torch", "gym", "gymnasium", "pandas", "polars", "sklearn"):
        assert forbidden not in source


def test_disk_adverse_builder_does_not_call_full_copy_helpers():
    source = Path("mmrt/execution/adverse_selection.py").read_text(encoding="utf-8")
    body = source.split("def build_adverse_selection_dataset_to_disk", 1)[1]
    forbidden = [
        "_valid_l2_view_from_tape(",
        "_valid_l2_views(",
        "_trade_flow_view_from_tape(",
        "_precompute_kyle_samples(",
        "_kyle_samples_for_disk_builder(",
        'np.asarray(events["local_ts_us"], dtype=np.int64)',
        'np.asarray(tape.arrays.events["local_ts_us"], dtype=np.int64)',
        "feature_rows.append",
        "label_rows.append",
        "mask_rows.append",
        "kept_features.append",
    ]
    offenders = [token for token in forbidden if token in body]
    assert offenders == []


def test_build_adverse_selection_signals_large_path_is_disk_backed():
    source = Path("mmrt/cli/build_adverse_selection_signals.py").read_text(encoding="utf-8")
    body = source.split("def build_adverse_selection_signals_from_config", 1)[1]
    assert "build_adverse_selection_features_to_disk" in body
    assert "build_adverse_selection_feature_dataset(" not in body


def test_adverse_streaming_fit_approx_auc_does_not_concatenate_scores():
    source = Path("mmrt/execution/adverse_selection_fit.py").read_text(encoding="utf-8")
    body = source.split("def fit_adverse_baselines_streaming", 1)[1]
    assert "BinaryHistogramAUC" in source
    assert "approx_histogram" in body
    if "np.concatenate" in body:
        assert 'metrics_mode == "exact"' in body or "metrics_mode == 'exact'" in body


def test_linear_audit_does_not_accumulate_full_column_lists():
    source = Path("mmrt/execution/linear_feature_audit.py").read_text(encoding="utf-8")
    assert "raw_cols" not in source
    assert "z_cols" not in source


def test_train_adverse_selection_progress_interval_is_wired():
    source = Path("mmrt/cli/train_adverse_selection.py").read_text(encoding="utf-8")
    assert "progress_interval" in source
    assert "progress_interval=config.progress_interval" in source


def test_build_adverse_selection_signals_progress_interval_is_wired():
    source = Path("mmrt/cli/build_adverse_selection_signals.py").read_text(encoding="utf-8")
    assert "--progress-interval" in source
    assert "progress_interval=config.progress_interval" in source


def test_build_adverse_selection_signals_large_path_does_not_construct_full_artifact():
    source = Path("mmrt/cli/build_adverse_selection_signals.py").read_text(encoding="utf-8")
    body = source.split("def build_adverse_selection_signals_from_config", 1)[1]
    assert "AdverseSelectionSignalArtifact(" not in body


def test_env_does_not_hard_gate_quotes_from_executable_edge():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")

    forbidden_mutations = (
        "quote.bid_enabled = False",
        "quote.ask_enabled = False",
        "quote = replace(quote",
        "quote_allowed",
    )
    offenders = [token for token in forbidden_mutations if token in source]
    assert offenders == []


def test_build_linear_signals_cli_has_no_rl_or_adverse_dependencies():
    source = Path("mmrt/cli/build_linear_signals.py").read_text(encoding="utf-8")
    for forbidden in ("mmrt.rl", "torch", "gym", "gymnasium", "adverse_selection", "adverse_signal"):
        assert forbidden not in source


def test_train_execution_ppo_default_linear_signal_filename_has_builder_cli():
    source = Path("mmrt/cli/build_linear_signals.py").read_text(encoding="utf-8")
    assert "LINEAR_SIGNALS_FILENAME" in source
    assert "build_linear_signal_artifact_npz_from_execution_feature_chunks" in source
    assert "save_linear_signal_artifact_npz" not in source


def test_execution_env_default_reset_uses_linear_signal_first_row():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")
    reset_body = source.split("def reset(", 1)[1].split("def step(", 1)[0]
    assert "self.linear_signals.decision_event_index[0]" in reset_body
    assert "start = 0" not in reset_body


def test_build_linear_signals_cli_does_not_recompute_predictions_for_summary():
    source = Path("mmrt/cli/build_linear_signals.py").read_text(encoding="utf-8")
    assert "predict_linear_heads_for_execution_features" not in source
    assert "linear_model_bundle_from_train_result" not in source
    assert "linear_preprocess_states_from_train_result" not in source
    assert "iter_execution_linear_feature_chunks" not in source
    assert "build_linear_signal_build_result" not in source


def test_linear_signal_builder_streaming_path_has_no_scan_replay():
    source = Path("mmrt/execution/linear_signal_builder.py").read_text(encoding="utf-8")
    body = source.split("def build_linear_signal_artifact_npz_from_execution_feature_chunks", 1)[1].split("__all__", 1)[0]
    assert body.count("iter_execution_linear_feature_chunks(") == 1
    assert "_scan_execution_linear_feature_chunks" not in source
    assert "NpyChunkWriter" in source


def test_deprecated_execution_linear_feature_dataset_has_no_chunk_list_vstack_path():
    source = Path("mmrt/execution/linear_signal_builder.py").read_text(encoding="utf-8")
    body = source.split("def build_execution_linear_feature_dataset", 1)[1].split("@dataclass", 1)[0]
    assert "DeprecationWarning" in body
    assert "chunks = list" not in body
    assert "np.vstack" not in body


def test_execution_clis_validate_linear_metadata_with_artifact_start_only():
    for path in (
        Path("mmrt/cli/train_execution_ppo.py"),
        Path("mmrt/cli/audit_execution_sim.py"),
        Path("mmrt/cli/evaluate_execution_policy.py"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "validate_linear_signals_for_execution_tape" in text
        assert "_effective_start_event_index" not in text


def test_env_computes_signal_end_before_reward_step():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")
    step_body = source.split("def step(", 1)[1].split("def _event_key_at_index", 1)[0]
    assert step_body.find("terminal_due_to_signal_end") < step_body.find("compute_reward_step(")


def test_execution_env_nonterminal_step_targets_next_linear_signal_row():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")
    step_body = source.split("def step(", 1)[1].split("def _event_key_at_index", 1)[0]

    assert "target_event_index" in step_body
    assert "self.linear_signals.decision_event_index[next_signal_row]" in step_body
    assert "_validate_next_signal_target" in source

    fallback_body = source.split("def _fallback", 1)[1] if "def _fallback" in source else ""
    old_token = "processed_any and processed_valid_l2 and event_local > decision_end_local_ts_us"
    if old_token in step_body:
        assert old_token in fallback_body


def test_build_execution_tape_cli_uses_streaming_writer_not_materialized_plan():
    source = Path("mmrt/cli/build_execution_tape.py").read_text(encoding="utf-8")
    body = source.split("def build_execution_tape_from_config", 1)[1].split("def load_reconstructed_l2_events", 1)[0]
    forbidden = (
        "l2_events, l2_stats = load_reconstructed_l2_events",
        "trades, trade_stats = load_trade_prints",
        "merge_execution_events(",
        "build_execution_tape_object(",
        "save_execution_tape(",
    )
    offenders = [token for token in forbidden if token in body]
    assert offenders == []
    assert "StreamingExecutionTapeWriter" in body
    assert "iter_merged_execution_events(" in body


def test_build_execution_tape_parser_accepts_repeated_input_flags_source_guard():
    source = Path("mmrt/cli/build_execution_tape.py").read_text(encoding="utf-8")
    assert 'action="append"' in source
    assert 'nargs="+"' in source


def test_large_tape_clis_use_shape_only_execution_tape_validation():
    paths = [
        Path("mmrt/cli/build_linear_signals.py"),
        Path("mmrt/cli/train_adverse_selection.py"),
        Path("mmrt/cli/build_adverse_selection_signals.py"),
        Path("mmrt/cli/audit_execution_sim.py"),
        Path("mmrt/cli/train_execution_ppo.py"),
        Path("mmrt/cli/evaluate_execution_policy.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "load_execution_tape(" in text:
            assert "ExecutionTapeValidationMode.SHAPE_ONLY" in text or 'validation_mode="shape_only"' in text


def test_streaming_execution_tape_writer_finalize_does_not_full_validate_by_default():
    source = Path("mmrt/execution/execution_tape_writer.py").read_text(encoding="utf-8")
    finalize_body = source.split("def finalize(", 1)[1].split("__all__", 1)[0]
    assert "self.config.validation_mode" in finalize_body
    assert "ExecutionTapeArrays(" not in finalize_body
