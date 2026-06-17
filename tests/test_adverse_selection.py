import inspect
import json
from pathlib import Path
from decimal import Decimal
from collections import deque

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, LatencyConfig, QueueModelMode, SymbolSpec, TradePrint
from mmrt.execution.event_merge import merge_execution_events
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.execution.execution_tape import build_execution_tape, save_execution_tape
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.decision_grid import save_decision_grid
from mmrt.execution.adverse_selection import (
    AdverseSelectionConfig,
    CounterfactualQuoteConfig,
    DEFAULT_QUOTE_CANDIDATES,
    KyleLambdaConfig,
    VPINConfig,
    VPINState,
    _AdverseLabelLayout,
    _CONSERVATIVE_NUMBA_AVAILABLE,
    _build_conservative_fill_index,
    _labels_for_decision,
    _labels_for_decision_batch_conservative,
    build_adverse_selection_dataset_to_disk,
    summarize_adverse_selection_dataset,
)
from mmrt.cli.train_adverse_selection import (
    AdverseSelectionTrainCLIConfig,
    _build_adverse_selection_config,
    _config_from_args,
    build_arg_parser,
    main,
    run_adverse_selection_training,
)
from tests.grid_helpers import decision_grid_for_tape
from tests.adverse_helpers import build_tiny_adverse_selection_dataset



def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures", symbol="BTCUSDT", mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC", quote_asset="USDT", margin_asset="USDT", contract_type="PERPETUAL", status="TRADING",
        tick_size=Decimal("0.1"), min_price=Decimal("0.1"), max_price=Decimal("1000000"),
        step_size=Decimal("0.001"), min_qty=Decimal("0.001"), max_qty=Decimal("100"), min_notional=Decimal("0"),
        allowed_order_types=("LIMIT",), allowed_time_in_force=("GTC", "GTX"),
    )

def _spec() -> SymbolSpec:
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=0.0,
    )


def _l2(
    *,
    seq: int,
    local_ts_us: int,
    bid_ticks=(1000, 999),
    bid_sizes=(1.0, 2.0),
    ask_ticks=(1002, 1003),
    ask_sizes=(1.0, 2.0),
) -> ReconstructedL2Event:
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=1,
        is_snapshot_batch=True,
        book_top=BookTop(
            local_ts_us=local_ts_us,
            best_bid_tick=bid_ticks[0],
            best_ask_tick=ask_ticks[0],
            best_bid_size=bid_sizes[0],
            best_ask_size=ask_sizes[0],
        ),
        bid_depth=len(bid_ticks),
        ask_depth=len(ask_ticks),
        book_snapshot=BookLevelSnapshot(
            local_ts_us=local_ts_us,
            bid_ticks=tuple(bid_ticks),
            bid_sizes=tuple(bid_sizes),
            ask_ticks=tuple(ask_ticks),
            ask_sizes=tuple(ask_sizes),
        ),
    )


def _trade(
    *,
    local_ts_us: int,
    side: AggressorSide,
    price_tick: int,
    amount: float,
    source_row: int,
) -> TradePrint:
    return TradePrint(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        side=side,
        price_tick=price_tick,
        amount=amount,
        source_row=source_row,
    )


def _tape(l2_events, trades):
    merged = merge_execution_events(l2_events, trades).events
    return build_execution_tape(
        symbol_spec=_spec(),
        symbol_rules=_rules(),
        l2_events=tuple(l2_events),
        trades=tuple(trades),
        merged_events=merged,
        book_depth=3,
        created_at_utc="2026-01-01T00:00:00Z",
    )


def _save_tape(tmp_path, tape):
    root = tmp_path / "tape"
    save_execution_tape(tape, root, overwrite=True)
    save_decision_grid(root / "decision_grid", decision_grid_for_tape(tape), overwrite=True)
    return root


def _tiny_tape_root(tmp_path, tape=None):
    return _save_tape(tmp_path, tape or _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], []))


def _label_value(dataset, name, row=0):
    return float(dataset.labels[row, dataset.label_names.index(name)])


def _label_mask(dataset, name, row=0):
    return bool(dataset.label_masks[row, dataset.label_names.index(name)])


_TEST_MAX_DECISIONS_BY_CONFIG_ID: dict[int, int | None] = {}


def _base_config(**kwargs):
    kwargs.pop("decision_interval_us", None)
    max_decisions = kwargs.pop("max_decisions", 1)
    params = dict(
        flow_windows_us=(200,),
        kyle=KyleLambdaConfig(sample_interval_us=100, response_horizon_us=100, windows_us=(200,), min_samples=1),
        quote=CounterfactualQuoteConfig(
            order_qty=1.0,
            fill_horizon_us=1_000_000,
            adverse_horizon_us=1_000_000,
            queue_model=QueueModelConfig(mode=QueueModelMode.BALANCED, l2_decrease_weight=1.0, trade_at_level_weight=1.0),
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0),
        ),
        drop_incomplete_horizon=True,
    )
    params.update(kwargs)
    config = AdverseSelectionConfig(**params)
    _TEST_MAX_DECISIONS_BY_CONFIG_ID[id(config)] = max_decisions
    return config


def build_adverse_selection_dataset(tape, *, config, tmp_path):
    max_rows = _TEST_MAX_DECISIONS_BY_CONFIG_ID.get(id(config), 1)
    return build_tiny_adverse_selection_dataset(tape, config=config, tmp_path=tmp_path, max_rows=max_rows)


def test_vpin_fractional_bucket_splitting():
    state = VPINState(VPINConfig(bucket_volume=10.0, num_buckets=3, min_completed_buckets=1), deque())
    state.update_trade(side_code=1, price_tick=1000, amount=25.0, tick_size=0.1)
    assert state.completed_bucket_count == 2
    assert list(state.completed_imbalances) == [10.0, 10.0]
    assert state.current_total_volume == pytest.approx(5.0)
    assert state.vpin() == pytest.approx(1.0)


def test_counterfactual_bid_fills_by_trade_at_level_after_queue_consumed(tmp_path):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=1_300_000, bid_ticks=(990, 989), ask_ticks=(992, 993))],
        [_trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1000, amount=2.0, source_row=0)],
    )
    dataset = build_adverse_selection_dataset(tape, config=_base_config(), tmp_path=tmp_path)
    assert dataset.num_decisions == 1
    assert _label_value(dataset, "bid_touch_filled") == 1.0
    assert _label_value(dataset, "bid_touch_fill_latency_us") > 0.0
    assert _label_mask(dataset, "bid_touch_adverse_bps") is True


def test_disappeared_visible_level_advances_queue_then_later_trade_fills(tmp_path):
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100),
            _l2(seq=1, local_ts_us=200, bid_ticks=(1001, 999), bid_sizes=(1.0, 2.0)),
            _l2(seq=2, local_ts_us=1_400_000),
        ],
        [_trade(local_ts_us=300, side=AggressorSide.SELL, price_tick=1000, amount=1.0, source_row=0)],
    )
    config = _base_config(
        quote=CounterfactualQuoteConfig(
            order_qty=1.0,
            fill_horizon_us=1_000_000,
            adverse_horizon_us=1_000_000,
            queue_model=QueueModelConfig(mode=QueueModelMode.BALANCED, l2_decrease_weight=1.0, trade_at_level_weight=1.0),
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0),
        )
    )
    dataset = build_adverse_selection_dataset(tape, config=config, tmp_path=tmp_path)
    assert _label_value(dataset, "bid_touch_filled") == 1.0
    assert _label_value(dataset, "bid_touch_fill_latency_us") == 200.0


def test_conservative_mode_does_not_advance_queue_on_l2_disappearance(tmp_path):
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100),
            _l2(seq=1, local_ts_us=200, bid_ticks=(999, 998), bid_sizes=(2.0, 2.0)),
            _l2(seq=2, local_ts_us=1_400_000),
        ],
        [_trade(local_ts_us=300, side=AggressorSide.SELL, price_tick=1000, amount=1.0, source_row=0)],
    )
    config = _base_config(
        quote=CounterfactualQuoteConfig(
            order_qty=1.0,
            fill_horizon_us=1_000_000,
            adverse_horizon_us=1_000_000,
            queue_model=QueueModelConfig(mode=QueueModelMode.CONSERVATIVE),
        )
    )
    dataset = build_adverse_selection_dataset(tape, config=config, tmp_path=tmp_path)
    assert _label_value(dataset, "bid_touch_filled") == 0.0


def test_label_masks_for_incomplete_horizon(tmp_path):
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    drop_dataset = build_adverse_selection_dataset(tape, config=_base_config(drop_incomplete_horizon=True), tmp_path=tmp_path)
    assert drop_dataset.num_decisions == 0

    keep_dataset = build_adverse_selection_dataset(tape, config=_base_config(drop_incomplete_horizon=False), tmp_path=tmp_path)
    assert keep_dataset.num_decisions == 1
    assert _label_mask(keep_dataset, "bid_touch_filled") is False
    assert _label_mask(keep_dataset, "ask_touch_filled") is False


def test_dataset_shape_and_summary(tmp_path):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=1_300_000)],
        [_trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1000, amount=2.0, source_row=0)],
    )
    dataset = build_adverse_selection_dataset(tape, config=_base_config(), tmp_path=tmp_path)
    assert dataset.features.dtype == np.float32
    assert dataset.labels.dtype == np.float32
    assert dataset.label_masks.dtype == np.bool_
    assert dataset.num_features == len(dataset.feature_names)
    assert dataset.num_labels == len(dataset.label_names)
    summary = summarize_adverse_selection_dataset(dataset)
    assert summary["num_decisions"] == dataset.num_decisions
    assert "vpin_mean" in summary["features"]


def _training_tape_with_multiple_decisions():
    l2_events = []
    for i, ts in enumerate(range(100, 3300, 100)):
        shift = (i % 3) - 1
        l2_events.append(
            _l2(
                seq=i,
                local_ts_us=ts,
                bid_ticks=(1000 + shift, 999 + shift),
                ask_ticks=(1002 + shift, 1003 + shift),
                bid_sizes=(1.0 + (i % 2) * 0.5, 2.0),
                ask_sizes=(1.0 + ((i + 1) % 2) * 0.5, 2.0),
            )
        )
    trades = [
        _trade(local_ts_us=150, side=AggressorSide.SELL, price_tick=999, amount=3.0, source_row=0),
        _trade(local_ts_us=250, side=AggressorSide.BUY, price_tick=1003, amount=3.0, source_row=1),
        _trade(local_ts_us=450, side=AggressorSide.SELL, price_tick=1000, amount=3.0, source_row=2),
        _trade(local_ts_us=650, side=AggressorSide.BUY, price_tick=1002, amount=3.0, source_row=3),
        _trade(local_ts_us=850, side=AggressorSide.SELL, price_tick=999, amount=3.0, source_row=4),
        _trade(local_ts_us=1050, side=AggressorSide.BUY, price_tick=1003, amount=3.0, source_row=5),
    ]
    return _tape(l2_events, trades)


def test_run_adverse_selection_training_writes_summary_and_model(tmp_path):
    tape_root = _tiny_tape_root(tmp_path, _training_tape_with_multiple_decisions())
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"

    summary = run_adverse_selection_training(
        AdverseSelectionTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            output_json=str(output_json),
            model_npz=str(model_npz),
            overwrite=True,
            fill_horizon_us=1_000,
            adverse_horizon_us=1_000,
            order_qty=1.0,
            train_fraction=0.6,
            min_train_samples=1,
            target_names=("bid_touch_filled", "ask_touch_filled", "bid_touch_toxic_cost_bps", "ask_touch_toxic_cost_bps"),
        )
    )

    assert output_json.exists()
    assert json.loads(output_json.read_text()) == summary
    assert summary["run_type"] == "train_adverse_selection"
    assert summary["dataset"]["num_decisions"] > 0
    assert summary["baseline"]["enabled"] is True
    if model_npz.exists():
        npz = np.load(model_npz, allow_pickle=True)
        assert str(npz["schema"]) == "mmrt_adverse_selection_ridge_grid_v1"
        assert "feature_mean" in npz
        assert "coefficients" in npz


def test_train_adverse_selection_profile_rows_writes_profile_only(tmp_path):
    tape_root = _tiny_tape_root(tmp_path, _training_tape_with_multiple_decisions())
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"
    profile_json = tmp_path / "profile.json"

    summary = run_adverse_selection_training(
        AdverseSelectionTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            output_json=str(output_json),
            model_npz=str(model_npz),
            overwrite=True,
            profile_rows=3,
            profile_output_json=str(profile_json),
            fill_horizon_us=1_000,
            adverse_horizon_us=1_000,
            order_qty=1.0,
            min_train_samples=1,
        )
    )

    assert summary["run_type"] == "profile_adverse_selection_label_generation"
    assert summary["decision_grid_rows_considered"] == 3
    assert summary["timing"]["rows_per_second"] > 0.0
    assert profile_json.exists()
    assert json.loads(profile_json.read_text()) == summary
    assert not output_json.exists()
    assert not model_npz.exists()


def test_adverse_dataset_reuses_complete_manifest_and_rejects_partial_root(tmp_path):
    tape = _training_tape_with_multiple_decisions()
    config = _base_config(
        quote=CounterfactualQuoteConfig(
            order_qty=1.0,
            fill_horizon_us=1_000,
            adverse_horizon_us=1_000,
            queue_model=QueueModelConfig(mode=QueueModelMode.CONSERVATIVE),
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0),
        ),
        max_decisions=4,
    )
    grid = decision_grid_for_tape(tape, max_rows=4)
    root = tmp_path / "ds"
    work_dir = tmp_path / "work"
    first = build_adverse_selection_dataset_to_disk(
        tape,
        config=config,
        decision_grid=grid,
        output_root=root,
        work_dir=work_dir,
        chunk_rows=4096,
        overwrite=True,
        cleanup_work_dir=False,
        label_engine="scalar",
    )
    second = build_adverse_selection_dataset_to_disk(
        tape,
        config=config,
        decision_grid=grid,
        output_root=root,
        work_dir=work_dir,
        chunk_rows=4096,
        overwrite=False,
        cleanup_work_dir=False,
        label_engine="scalar",
    )
    assert second.manifest.created_at_utc == first.manifest.created_at_utc

    partial_root = tmp_path / "partial_ds"
    partial_root.mkdir()
    with pytest.raises(FileExistsError, match="partial adverse dataset root"):
        build_adverse_selection_dataset_to_disk(
            tape,
            config=config,
            decision_grid=grid,
            output_root=partial_root,
            work_dir=work_dir,
            chunk_rows=4096,
            overwrite=False,
            cleanup_work_dir=False,
            label_engine="scalar",
        )


def test_adverse_selection_all_unknown_targets_preserves_skip_reasons(tmp_path):
    tape_root = _tiny_tape_root(tmp_path, _training_tape_with_multiple_decisions())
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"

    summary = run_adverse_selection_training(
        AdverseSelectionTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            output_json=str(output_json),
            model_npz=str(model_npz),
            overwrite=True,
            fill_horizon_us=1_000,
            adverse_horizon_us=1_000,
            order_qty=1.0,
            min_train_samples=1,
            target_names=("not_a_label", "also_not_a_label"),
        )
    )

    assert summary["status"] == "warning"
    assert summary["model_npz"] is None
    assert not model_npz.exists()

    baseline = summary["baseline"]
    assert baseline["skipped"] is True
    assert baseline["skip_reason"] == "all_targets_skipped"
    assert baseline["fitted_target_count"] == 0
    assert baseline["requested_target_count"] == 2

    assert set(baseline["targets"]) == {"not_a_label", "also_not_a_label"}
    assert baseline["targets"]["not_a_label"]["skipped"] is True
    assert baseline["targets"]["not_a_label"]["skip_reason"] == "unknown_target"
    assert baseline["targets"]["also_not_a_label"]["skip_reason"] == "unknown_target"


def test_adverse_selection_not_enough_decisions_preserves_target_skip_reasons(tmp_path):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    tape_root = _tiny_tape_root(tmp_path, tape)
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"

    summary = run_adverse_selection_training(
        AdverseSelectionTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            output_json=str(output_json),
            model_npz=str(model_npz),
            overwrite=True,
            fill_horizon_us=1_000,
            adverse_horizon_us=1_000,
            order_qty=1.0,
            drop_incomplete_horizon=False,
            target_names=("bid_touch_filled", "ask_touch_filled"),
        )
    )

    assert summary["status"] == "warning"
    assert summary["model_npz"] is None
    assert not model_npz.exists()

    baseline = summary["baseline"]
    assert baseline["skipped"] is True
    assert baseline["skip_reason"] == "all_targets_skipped"
    assert baseline["fitted_target_count"] == 0
    assert baseline["requested_target_count"] == 2
    assert set(baseline["targets"]) == {"bid_touch_filled", "ask_touch_filled"}
    assert baseline["targets"]["bid_touch_filled"]["skip_reason"] == "not_enough_train_rows"
    assert baseline["targets"]["ask_touch_filled"]["skip_reason"] == "not_enough_train_rows"


def test_train_adverse_selection_main_writes_summary_and_prints_json(tmp_path, capsys):
    tape_root = _tiny_tape_root(tmp_path, _training_tape_with_multiple_decisions())
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"
    rc = main([
        "--tape-root", str(tape_root),
        "--decision-grid", str(tape_root / "decision_grid"),
        "--output-json", str(output_json),
        "--model-npz", str(model_npz),
        "--overwrite",
        "--fill-horizon-us", "1000",
        "--adverse-horizon-us", "1000",
        "--order-qty", "1.0",
        "--min-train-samples", "1",
    ])
    assert rc == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    disk_summary = json.loads(output_json.read_text())
    assert stdout_summary == disk_summary


def test_train_adverse_selection_overwrite_guard(tmp_path):
    tape_root = tmp_path / "missing_tape"
    output_json = tmp_path / "summary.json"
    model_npz = tmp_path / "model.npz"
    output_json.write_text("{}")
    model_npz.write_bytes(b"exists")
    with pytest.raises(FileExistsError):
        run_adverse_selection_training(
            AdverseSelectionTrainCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                output_json=str(output_json),
                model_npz=str(model_npz),
            )
        )


def test_quote_candidate_parser_rejects_malformed_offsets():
    with pytest.raises(ValueError, match="malformed quote candidate"):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", quote_candidates="inside_x")
    with pytest.raises(ValueError, match="malformed quote candidate"):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", quote_candidates="away_0")


def test_quote_candidate_parser_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate quote candidate"):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", quote_candidates="touch,touch")


def test_quote_candidate_parser_validates_sequence_values():
    with pytest.raises(ValueError, match="QuoteCandidateConfig"):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", quote_candidates=("touch",))  # type: ignore[arg-type]


def test_train_adverse_selection_config_wires_latency_to_counterfactual_config():
    cfg = AdverseSelectionTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/grid",
        decision_compute_latency_us=7,
        order_entry_latency_us=11,
    )
    adverse_cfg = _build_adverse_selection_config(cfg)
    assert adverse_cfg.quote.latency_config.decision_compute_latency_us == 7
    assert adverse_cfg.quote.latency_config.order_entry_latency_us == 11


def test_train_adverse_selection_parser_accepts_latency_args():
    args = build_arg_parser().parse_args([
        "--tape-root",
        "/tmp/tape",
        "--decision-grid",
        "/tmp/grid",
        "--decision-compute-latency-us",
        "7",
        "--order-entry-latency-us",
        "11",
    ])
    cfg = _config_from_args(args)
    assert cfg.decision_compute_latency_us == 7
    assert cfg.order_entry_latency_us == 11


def _conservative_label_parity_fixture(tmp_path):
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100, bid_ticks=(1000, 999), ask_ticks=(1004, 1005), bid_sizes=(1.0, 2.0), ask_sizes=(1.0, 2.0)),
            _l2(seq=1, local_ts_us=650, bid_ticks=(1000, 999), ask_ticks=(1001, 1002), bid_sizes=(1.0, 2.0), ask_sizes=(1.0, 2.0)),
            _l2(seq=2, local_ts_us=800, bid_ticks=(1001, 1000), ask_ticks=(1003, 1004), bid_sizes=(1.0, 2.0), ask_sizes=(1.0, 2.0)),
            _l2(seq=3, local_ts_us=1_200_000, bid_ticks=(990, 989), ask_ticks=(992, 993), bid_sizes=(1.0, 2.0), ask_sizes=(1.0, 2.0)),
            _l2(seq=4, local_ts_us=2_000_000, bid_ticks=(980, 979), ask_ticks=(982, 983), bid_sizes=(1.0, 2.0), ask_sizes=(1.0, 2.0)),
        ],
        [
            _trade(local_ts_us=700, side=AggressorSide.SELL, price_tick=1000, amount=2.5, source_row=0),
            _trade(local_ts_us=720, side=AggressorSide.BUY, price_tick=1004, amount=2.5, source_row=1),
            _trade(local_ts_us=900, side=AggressorSide.SELL, price_tick=998, amount=1.0, source_row=2),
            _trade(local_ts_us=920, side=AggressorSide.BUY, price_tick=1005, amount=1.0, source_row=3),
        ],
    )
    config = _base_config(
        quote=CounterfactualQuoteConfig(
            quote_candidates=DEFAULT_QUOTE_CANDIDATES,
            order_qty=1.0,
            fill_horizon_us=1_000_000,
            adverse_horizon_us=1_000_000,
            toxic_threshold_bps=0.1,
            queue_model=QueueModelConfig(mode=QueueModelMode.CONSERVATIVE, trade_at_level_weight=1.0),
            latency_config=LatencyConfig(decision_compute_latency_us=250, order_entry_latency_us=250),
        ),
        max_decisions=4,
    )
    grid = decision_grid_for_tape(tape, max_rows=4)
    from mmrt.execution.adverse_selection_index import AdverseSelectionIndexConfig, build_or_load_adverse_selection_index

    index = build_or_load_adverse_selection_index(
        tape,
        config=AdverseSelectionIndexConfig(
            output_root=str(tmp_path / "idx"),
            kyle=config.kyle,
            use_notional_flow=config.kyle.use_notional_flow,
            tick_size=tape.manifest.symbol_spec.tick_size,
            chunk_rows=4096,
            overwrite=True,
        ),
    )
    return tape, config, grid, index


def test_conservative_batch_labels_match_scalar_reference(tmp_path):
    tape, config, grid, index = _conservative_label_parity_fixture(tmp_path)
    layout = _AdverseLabelLayout.from_config(config)
    fill_index = _build_conservative_fill_index(tape)
    events_ts = tape.arrays.events["local_ts_us"]
    scalar_labels = []
    scalar_masks = []
    scalar_keep = []
    for row in range(grid.n_rows):
        out = _labels_for_decision(
            tape,
            config=config,
            layout=layout,
            last_event_local_ts_us=int(events_ts[-1]),
            events_local_ts_us=events_ts,
            decision_event_index=int(grid.decision_event_index[row]),
            latest_book_ptr=int(grid.book_ptr[row]),
            decision_key=EventKey(int(grid.decision_local_ts_us[row]), int(grid.decision_event_seq[row])),
            future_mid_lookup=index.valid_l2,
            fill_index=fill_index,
        )
        scalar_keep.append(out is not None)
        if out is None:
            scalar_labels.append([np.nan] * layout.label_count)
            scalar_masks.append([False] * layout.label_count)
        else:
            scalar_labels.append(out[0])
            scalar_masks.append(out[1])

    labels, masks, keep_rows, backend = _labels_for_decision_batch_conservative(
        tape,
        config=config,
        layout=layout,
        last_event_local_ts_us=int(events_ts[-1]),
        events_local_ts_us=events_ts,
        decision_event_index=grid.decision_event_index,
        latest_book_ptr=grid.book_ptr,
        decision_local_ts_us=grid.decision_local_ts_us,
        decision_event_seq=grid.decision_event_seq,
        future_mid_lookup=index.valid_l2,
        fill_index=fill_index,
        label_engine="scalar",
    )
    assert backend == "scalar"
    np.testing.assert_array_equal(keep_rows, np.asarray(scalar_keep, dtype=np.bool_))
    np.testing.assert_allclose(labels, np.asarray(scalar_labels, dtype=np.float32), equal_nan=True)
    np.testing.assert_array_equal(masks, np.asarray(scalar_masks, dtype=np.bool_))
    label_index = {name: i for i, name in enumerate(layout.label_names)}
    assert masks[:, label_index["bid_touch_fill_latency_us"]].any()
    assert masks[:, label_index["ask_touch_fill_latency_us"]].any()
    assert masks[:, label_index["bid_touch_adverse_bps"]].any()
    assert masks[:, label_index["ask_touch_toxic_fill"]].any()
    assert masks[:, label_index["bid_away_1_toxic_cost_bps"]].any()


def test_conservative_numba_backend_matches_or_errors_clearly(tmp_path):
    tape, config, grid, index = _conservative_label_parity_fixture(tmp_path)
    layout = _AdverseLabelLayout.from_config(config)
    fill_index = _build_conservative_fill_index(tape)
    events_ts = tape.arrays.events["local_ts_us"]
    kwargs = dict(
        tape=tape,
        config=config,
        layout=layout,
        last_event_local_ts_us=int(events_ts[-1]),
        events_local_ts_us=events_ts,
        decision_event_index=grid.decision_event_index,
        latest_book_ptr=grid.book_ptr,
        decision_local_ts_us=grid.decision_local_ts_us,
        decision_event_seq=grid.decision_event_seq,
        future_mid_lookup=index.valid_l2,
        fill_index=fill_index,
    )
    if not _CONSERVATIVE_NUMBA_AVAILABLE:
        with pytest.raises(ValueError, match="requires numba"):
            _labels_for_decision_batch_conservative(**kwargs, label_engine="numba")
        return
    scalar = _labels_for_decision_batch_conservative(**kwargs, label_engine="scalar")
    accelerated = _labels_for_decision_batch_conservative(**kwargs, label_engine="numba")
    assert accelerated[3] == "numba"
    np.testing.assert_allclose(accelerated[0], scalar[0], equal_nan=True)
    np.testing.assert_array_equal(accelerated[1], scalar[1])
    np.testing.assert_array_equal(accelerated[2], scalar[2])


def test_adverse_selection_schema_constant_is_direct_string():
    source = Path("mmrt/execution/adverse_signal.py").read_text(encoding="utf-8")
    assert 'ADVERSE_SELECTION_MODEL_SCHEMA = "mmrt_adverse_selection_ridge_grid_v1"' in source
    assert '"mmrt_adverse_selection_ridge" + "_" + "v" + "2"' not in source


def test_config_parses_windows_queue_mode_and_targets():
    cfg = AdverseSelectionTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/grid",
        flow_windows_us="100,200",
        kyle_windows_us="1000,2000",
        queue_mode="balanced",
        label_engine="scalar",
        profile_rows=100,
        target_names="bid_touch_filled,ask_touch_filled",
    )
    assert cfg.flow_windows_us == (100, 200)
    assert cfg.kyle_windows_us == (1000, 2000)
    assert cfg.queue_mode == QueueModelMode.BALANCED
    assert cfg.label_engine == "scalar"
    assert cfg.profile_rows == 100
    assert cfg.target_names == ("bid_touch_filled", "ask_touch_filled")
    with pytest.raises(ValueError):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", train_fraction=1.0)
    with pytest.raises(ValueError):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", l2_decrease_weight=1.1)
    with pytest.raises(ValueError):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", target_names="bid_touch_filled,")
    with pytest.raises(ValueError):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", order_entry_latency_us=-1)
    with pytest.raises(ValueError):
        AdverseSelectionTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/grid", label_engine="bad")


def test_adverse_selection_modules_do_not_import_forbidden_layers():
    import mmrt.execution.adverse_selection as adverse
    import mmrt.cli.train_adverse_selection as cli

    adverse_source = inspect.getsource(adverse)
    cli_source = inspect.getsource(cli)

    for forbidden in (
        "argparse",
        "torch",
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.rl",
        "mmrt.cli",
        "mmrt.storage",
        "mmrt.linear",
        "load_execution_tape",
    ):
        assert forbidden not in adverse_source
    for forbidden in (
        "torch",
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.rl",
        "mmrt.storage",
        "mmrt.linear",
    ):
        assert forbidden not in cli_source

from mmrt.execution.adverse_selection_index import ValidL2Index
from mmrt.time_key import EventKey, MAX_EVENT_SEQ


def test_kyle_future_mid_uses_last_l2_at_same_timestamp():
    index = ValidL2Index(
        local_ts_us=np.asarray([100, 200, 200], dtype=np.int64),
        event_seq=np.asarray([0, 1, 2], dtype=np.int64),
        mid_tick=np.asarray([1001.0, 1011.0, 1021.0], dtype=np.float32),
    )
    assert index.future_mid_tick_at_or_after(EventKey(200, MAX_EVENT_SEQ)) == pytest.approx(1021.0)
    assert index.future_mid_tick_at_or_after(EventKey(200, 1)) == pytest.approx(1011.0)


def test_kyle_samples_become_ready_for_dataset_features(tmp_path):
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100),
            _l2(seq=1, local_ts_us=200, bid_ticks=(1001, 1000), ask_ticks=(1003, 1004)),
            _l2(seq=2, local_ts_us=300, bid_ticks=(1002, 1001), ask_ticks=(1004, 1005)),
        ],
        [_trade(local_ts_us=150, side=AggressorSide.BUY, price_tick=1002, amount=1.0, source_row=0)],
    )
    config = _base_config(
        decision_interval_us=100,
        max_decisions=2,
        kyle=KyleLambdaConfig(sample_interval_us=50, response_horizon_us=100, windows_us=(1_000,), min_samples=1),
        quote=CounterfactualQuoteConfig(
            order_qty=1.0,
            fill_horizon_us=100,
            adverse_horizon_us=100,
            queue_model=QueueModelConfig(mode=QueueModelMode.CONSERVATIVE),
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0),
        ),
        drop_incomplete_horizon=False,
    )
    dataset = build_adverse_selection_dataset(tape, config=config, tmp_path=tmp_path)
    idx = dataset.feature_names.index("kyle_n_1ms")
    assert dataset.num_decisions >= 2
    assert dataset.features[-1, idx] >= 1.0


def test_labels_for_decision_does_not_recompute_label_names():
    source = Path("mmrt/execution/adverse_selection.py").read_text()
    body = source.split("def _labels_for_decision", 1)[1].split("class _AdverseFeatureRow", 1)[0]
    assert "adverse_selection_label_names(config)" not in body


def test_counterfactual_fill_uses_precomputed_end_event_index():
    source = Path("mmrt/execution/adverse_selection.py").read_text()
    assert "end_event_index" in source
    assert "np.searchsorted(events_local_ts_us" in source
