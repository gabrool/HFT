from pathlib import Path
import json

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActionSpec,
    BookLevelSnapshot,
    BookTop,
    FillReason,
    OrderSide,
    PositionState,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, load_execution_tape, save_execution_tape
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    LINEAR_SIGNALS_FILENAME,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    predictions_to_signal_arrays,
    save_linear_signal_artifact_npz,
)
from mmrt.execution.metrics import ExecutionMetricAccumulator, summarize_execution_steps
from mmrt.execution.diagnostics import ExecutionDiagnosticsConfig, diagnose_execution_metrics
from mmrt.cli.audit_execution_sim import (
    ExecutionSimAuditConfig,
    run_execution_sim_audit,
    main,
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
    top = BookTop(
        local_ts_us=local_ts_us,
        best_bid_tick=bid_ticks[0],
        best_ask_tick=ask_ticks[0],
        best_bid_size=bid_sizes[0],
        best_ask_size=ask_sizes[0],
    )
    snapshot = BookLevelSnapshot(
        local_ts_us=local_ts_us,
        bid_ticks=tuple(bid_ticks),
        bid_sizes=tuple(bid_sizes),
        ask_ticks=tuple(ask_ticks),
        ask_sizes=tuple(ask_sizes),
    )
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us - 10,
        max_ts_us=local_ts_us - 5,
        num_updates=1,
        is_snapshot_batch=(seq == 0),
        book_top=top,
        bid_depth=len(bid_ticks),
        ask_depth=len(ask_ticks),
        book_snapshot=snapshot,
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
        ts_us=local_ts_us - 1,
        side=side,
        price_tick=price_tick,
        amount=amount,
        trade_id=str(source_row),
        source_row=source_row,
    )


def _tape(l2_events, trades):
    plan = merge_execution_events(l2_events, trades)
    return build_execution_tape(
        symbol_spec=_spec(),
        l2_events=l2_events,
        trades=trades,
        merged_events=plan.events,
        book_depth=2,
    )


def _save_tape(tmp_path, tape):
    root = tmp_path / "execution_tape"
    save_execution_tape(tape, root, overwrite=True)
    return root


def _signal_arrays(n_rows: int = 16):
    return predictions_to_signal_arrays({
        NO_MOVE_PROBA_KEY: np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n_rows, 1)),
        DIRECTION_PROBA_KEY: np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (n_rows, 1)),
        MAGNITUDE_UP_KEY: np.full(n_rows, np.log1p(10.0), dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.full(n_rows, np.log1p(5.0), dtype=np.float32),
    })


def _linear_artifact_for_tape(tape, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0):
    arrays = _signal_arrays(n_rows)
    pairs = []
    for event_index, event in enumerate(tape.arrays.events):
        if event_index < start_event_index:
            continue
        if int(event["event_type_code"]) != 1:
            continue
        book_ptr = int(event["book_ptr"])
        if book_ptr >= 0:
            pairs.append((event_index, int(tape.arrays.l2_events[book_ptr]["local_ts_us"])))
    if not pairs:
        pairs.append((start_event_index, int(tape.manifest.start_local_ts_us)))
    decision_event_index = [pair[0] for pair in pairs[:n_rows]]
    decision_local_ts_us = [pair[1] for pair in pairs[:n_rows]]
    while len(decision_event_index) < n_rows:
        decision_event_index.append(decision_event_index[-1] + 1)
        decision_local_ts_us.append(decision_local_ts_us[-1] + decision_interval_us)
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_interval_us=decision_interval_us,
        start_event_index=start_event_index,
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.asarray(decision_event_index, dtype=np.int64),
        decision_local_ts_us=np.asarray(decision_local_ts_us, dtype=np.int64),
    )


def _save_linear_signals(root, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0):
    tape = load_execution_tape(root)
    artifact = _linear_artifact_for_tape(
        tape, n_rows=n_rows, decision_interval_us=decision_interval_us, start_event_index=start_event_index
    )
    path = root / LINEAR_SIGNALS_FILENAME
    save_linear_signal_artifact_npz(path, artifact, overwrite=True)
    return path


def test_metrics_empty_summary():
    acc = ExecutionMetricAccumulator()
    summary = acc.as_dict()

    assert summary["steps"]["count"] == 0
    assert summary["fills"]["count"] == 0
    assert summary["rewards"]["total_raw"] == 0.0


def test_disabled_audit_runs_and_warns_no_fills(tmp_path):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    tape_root = _save_tape(tmp_path, tape)
    _save_linear_signals(tape_root)
    output_json = tmp_path / "summary.json"

    summary = run_execution_sim_audit(
        ExecutionSimAuditConfig(
            tape_root=str(tape_root),
            output_json=str(output_json),
            policy="disabled",
            max_steps=2,
            decision_interval_us=50,
            overwrite=True,
        )
    )

    assert output_json.exists()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded == summary

    assert summary["audit_type"] == "execution_sim"
    assert summary["metrics"]["steps"]["count"] >= 1
    assert summary["metrics"]["fills"]["count"] == 0
    assert "no_fills_observed" in summary["diagnostics"]["warnings"]


def test_bid_audit_records_trade_fill_and_reward(tmp_path):
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=300),
    ]
    trades = [
        _trade(
            local_ts_us=150,
            side=AggressorSide.SELL,
            price_tick=1000,
            amount=2.0,
            source_row=0,
        )
    ]
    tape_root = _save_tape(tmp_path, _tape(l2_events, trades))
    _save_linear_signals(tape_root, decision_interval_us=250)

    summary = run_execution_sim_audit(
        ExecutionSimAuditConfig(
            tape_root=str(tape_root),
            output_json=str(tmp_path / "summary.json"),
            policy="bid",
            max_steps=2,
            decision_interval_us=250,
            max_order_qty=1.0,
            default_order_qty=1.0,
            decision_compute_latency_us=0,
            order_entry_latency_us=0,
            cancel_latency_us=0,
            overwrite=True,
        )
    )

    metrics = summary["metrics"]
    assert metrics["fills"]["count"] == 1
    assert metrics["fills"]["buy_count"] == 1
    assert metrics["fills"]["reason_counts"][FillReason.TRADE_THROUGH.value] == 1
    assert metrics["position"]["final_inventory_qty"] == pytest.approx(1.0)
    assert metrics["rewards"]["total_raw"] == pytest.approx(0.005005)


def test_diagnostics_errors_on_zero_steps():
    metrics = ExecutionMetricAccumulator().as_dict()
    report = diagnose_execution_metrics(metrics)

    assert report.status == "error"
    assert "zero_steps" in report.errors


def test_diagnostics_threshold_warnings():
    metrics = ExecutionMetricAccumulator().as_dict()
    metrics["steps"]["count"] = 10
    metrics["steps"]["events_processed_total"] = 10
    metrics["steps"]["terminal_count"] = 1
    metrics["orders"]["cancel_rate_per_step"] = 2.0
    metrics["position"]["max_abs_inventory_qty"] = 5.0
    metrics["equity"]["max_drawdown"] = 3.0
    metrics["rewards"]["total_raw"] = -10.0

    report = diagnose_execution_metrics(
        metrics,
        config=ExecutionDiagnosticsConfig(
            max_cancel_rate_warn=1.0,
            max_abs_inventory_qty_warn=1.0,
            max_drawdown_warn=1.0,
            min_total_reward_warn=0.0,
            warn_if_no_fills=False,
            warn_if_no_turnover=False,
            warn_if_all_quotes_disabled=False,
        ),
    )

    assert report.status == "warning"
    assert "high_cancel_rate" in report.warnings
    assert "max_abs_inventory_qty_exceeded" in report.warnings
    assert "max_drawdown_exceeded" in report.warnings
    assert "total_reward_below_threshold" in report.warnings



def test_audit_execution_sim_requires_linear_signals_file(tmp_path):
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    tape_root = _save_tape(tmp_path, tape)
    with pytest.raises(FileNotFoundError):
        run_execution_sim_audit(
            ExecutionSimAuditConfig(
                tape_root=str(tape_root),
                output_json=str(tmp_path / "summary.json"),
                policy="disabled",
                max_steps=1,
                overwrite=True,
            )
        )

def test_audit_execution_sim_main_writes_summary_and_prints_json(tmp_path, capsys):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    tape_root = _save_tape(tmp_path, tape)
    _save_linear_signals(tape_root)
    output_json = tmp_path / "summary.json"

    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--output-json",
            str(output_json),
            "--policy",
            "disabled",
            "--max-steps",
            "1",
            "--decision-interval-us",
            "50",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert output_json.exists()

    stdout_payload = json.loads(capsys.readouterr().out)
    disk_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert stdout_payload == disk_payload
    assert stdout_payload["audit_type"] == "execution_sim"


def test_audit_execution_sim_refuses_overwrite_without_flag(tmp_path):
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    tape_root = _save_tape(tmp_path, tape)
    _save_linear_signals(tape_root)
    output_json = tmp_path / "summary.json"
    output_json.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_execution_sim_audit(
            ExecutionSimAuditConfig(
                tape_root=str(tape_root),
                output_json=str(output_json),
                policy="disabled",
            )
        )


def test_execution_sim_audit_config_validation():
    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="", policy="disabled")

    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="x", policy="bad")

    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="x", max_steps=0)

    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="x", queue_mode="bad")


def test_audit_execution_sim_accepts_zero_queue_weights():
    cfg = ExecutionSimAuditConfig(
        tape_root="/tmp/tape",
        l2_decrease_weight=0.0,
        trade_at_level_weight=0.0,
    )

    assert cfg.l2_decrease_weight == 0.0
    assert cfg.trade_at_level_weight == 0.0


def test_audit_execution_sim_rejects_queue_weights_above_one():
    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="/tmp/tape", l2_decrease_weight=1.1)

    with pytest.raises(ValueError):
        ExecutionSimAuditConfig(tape_root="/tmp/tape", trade_at_level_weight=1.1)


def test_audit_modules_have_no_forbidden_imports():
    paths = [
        Path("mmrt/execution/metrics.py"),
        Path("mmrt/execution/diagnostics.py"),
        Path("mmrt/cli/audit_execution_sim.py"),
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "import gym" not in source
        assert "import gymnasium" not in source
        assert "import pandas" not in source
        assert "import polars" not in source
        assert "import sklearn" not in source
        assert "import pyarrow" not in source
        assert "mmrt.storage" not in source
        assert "mmrt.linear.models" not in source
        assert "mmrt.rl" not in source

    metrics_source = Path("mmrt/execution/metrics.py").read_text(encoding="utf-8")
    diagnostics_source = Path("mmrt/execution/diagnostics.py").read_text(encoding="utf-8")

    assert "mmrt.execution.env" not in metrics_source
    assert "mmrt.execution.execution_tape" not in metrics_source
    assert "mmrt.cli" not in metrics_source

    assert "mmrt.execution.env" not in diagnostics_source
    assert "mmrt.execution.execution_tape" not in diagnostics_source
    assert "mmrt.cli" not in diagnostics_source
