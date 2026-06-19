"""Audit an existing execution tape by running a deterministic execution simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mmrt.execution.contracts import ActionSpec, LatencyConfig, PositionState, QueueModelMode
from mmrt.execution.diagnostics import ExecutionDiagnosticsConfig, diagnose_execution_metrics
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.horizon_diagnostics import (
    DEFAULT_HORIZONS_US,
    HorizonDiagnosticsAccumulator,
    HorizonDiagnosticsConfig,
    parse_horizon_diagnostics_us,
)
from mmrt.execution.adverse_runtime import AdverseRuntimeConfig
from mmrt.execution.adverse_signal import load_adverse_selection_signals
from mmrt.cli.execution_env_config import build_execution_env_config_from_attrs
from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    load_linear_signal_artifact_npz,
    linear_signal_artifact_summary,
)
from mmrt.execution.metrics import ExecutionMetricAccumulator
from mmrt.execution.quote_geometry import QuoteAction
from mmrt.cli.output import (
    STDOUT_MODES,
    compact_audit_summary,
    compact_json_line,
    print_human_summary,
    validate_stdout_mode,
    write_json_atomic,
)

AUDIT_POLICIES = (
    "disabled",
    "bid",
    "ask",
    "two_sided",
    "alternate_bid_ask",
)


def _policy_action(policy: str, step_index: int, *, action_size_raw: float) -> QuoteAction:
    if policy not in AUDIT_POLICIES:
        raise ValueError(f"policy must be one of {AUDIT_POLICIES}")
    if policy == "disabled":
        bid_enabled, ask_enabled = False, False
    elif policy == "bid":
        bid_enabled, ask_enabled = True, False
    elif policy == "ask":
        bid_enabled, ask_enabled = False, True
    elif policy == "two_sided":
        bid_enabled, ask_enabled = True, True
    else:
        bid_enabled, ask_enabled = (True, False) if step_index % 2 == 0 else (False, True)
    return QuoteAction(
        bid_enabled=bid_enabled,
        ask_enabled=ask_enabled,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=action_size_raw,
        ask_size_raw=action_size_raw,
    )


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite float")
    return out


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _coerce_queue_mode(value: QueueModelMode | str) -> QueueModelMode:
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"queue_mode has invalid value {value!r}") from exc
    raise ValueError("queue_mode must be QueueModelMode or str")


def _summary_config(config: "ExecutionSimAuditConfig") -> dict[str, object]:
    return {
        "decision_grid_path": config.decision_grid_path,
        "linear_signals_npz": config.linear_signals_npz,
        "adverse_signals_npz": config.adverse_signals_npz,
        "debug_output_json": config.debug_output_json,
        "horizon_debug_json": config.horizon_debug_json,
        "policy": config.policy,
        "max_steps": config.max_steps,
        "start_event_index": config.start_event_index,
        "cancel_guard_ticks": config.cancel_guard_ticks,
        "mmap_mode": config.mmap_mode,
        "max_distance_ticks": config.max_distance_ticks,
        "max_order_qty": config.max_order_qty,
        "post_only_gap_ticks": config.post_only_gap_ticks,
        "default_order_qty": config.default_order_qty,
        "action_size_raw": config.action_size_raw,
        "queue_mode": config.queue_mode.value,
        "l2_decrease_weight": config.l2_decrease_weight,
        "trade_at_level_weight": config.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.unknown_level_queue_ahead_qty,
        "dedupe_l2_decrease_with_trade_prints": config.dedupe_l2_decrease_with_trade_prints,
        "maker_fee_bps": config.maker_fee_bps,
        "edge_min_executable_edge_bps": config.edge_min_executable_edge_bps,
        "edge_latency_buffer_bps": config.edge_latency_buffer_bps,
        "edge_inventory_skew_bps_per_unit": config.edge_inventory_skew_bps_per_unit,
        "decision_compute_latency_us": config.decision_compute_latency_us,
        "order_entry_latency_us": config.order_entry_latency_us,
        "cancel_latency_us": config.cancel_latency_us,
        "inventory_penalty_bps": config.inventory_penalty_bps,
        "turnover_penalty_bps": config.turnover_penalty_bps,
        "cancel_penalty": config.cancel_penalty,
        "drawdown_penalty_rate": config.drawdown_penalty_rate,
        "terminal_inventory_penalty_bps": config.terminal_inventory_penalty_bps,
        "reward_scale": config.reward_scale,
        "horizon_diagnostics_enabled": config.horizon_diagnostics_enabled,
        "horizon_diagnostics_us": list(config.horizon_diagnostics_us),
        "stdout_mode": config.stdout_mode,
    }


@dataclass(frozen=True, slots=True)
class ExecutionSimAuditConfig:
    tape_root: str
    decision_grid_path: str
    output_json: str | None = None
    debug_output_json: str | None = None
    horizon_debug_json: str | None = None
    linear_signals_npz: str | None = None
    adverse_signals_npz: str | None = None
    overwrite: bool = False

    policy: str = "alternate_bid_ask"
    max_steps: int = 1_000
    start_event_index: int | None = None
    cancel_guard_ticks: int = 2
    mmap_mode: str | None = "r"

    max_distance_ticks: int = 1
    max_order_qty: float = 0.001
    post_only_gap_ticks: int = 1
    default_order_qty: float = 0.001
    action_size_raw: float = 100.0

    queue_mode: QueueModelMode | str = QueueModelMode.CONSERVATIVE
    l2_decrease_weight: float = 0.25
    trade_at_level_weight: float = 0.5
    unknown_level_queue_ahead_qty: float = 1_000_000_000.0
    dedupe_l2_decrease_with_trade_prints: bool = True

    maker_fee_bps: float = -0.5
    edge_min_executable_edge_bps: float = 0.0
    edge_latency_buffer_bps: float = 0.0
    edge_inventory_skew_bps_per_unit: float = 0.0

    decision_compute_latency_us: int = 50
    order_entry_latency_us: int = 500
    cancel_latency_us: int = 500

    inventory_penalty_bps: float = 0.0
    turnover_penalty_bps: float = 0.0
    cancel_penalty: float = 0.0
    drawdown_penalty_rate: float = 0.0
    terminal_inventory_penalty_bps: float = 0.0
    reward_scale: float = 1.0
    horizon_diagnostics_enabled: bool = True
    horizon_diagnostics_us: tuple[int, ...] = DEFAULT_HORIZONS_US
    stdout_mode: str = "summary"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        _require_nonempty_str(self.decision_grid_path, "decision_grid_path")
        if self.output_json is not None:
            _require_nonempty_str(self.output_json, "output_json")
        if self.debug_output_json is not None:
            _require_nonempty_str(self.debug_output_json, "debug_output_json")
        if self.horizon_debug_json is not None:
            _require_nonempty_str(self.horizon_debug_json, "horizon_debug_json")
        if self.linear_signals_npz is not None:
            _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        if self.adverse_signals_npz is not None:
            _require_nonempty_str(self.adverse_signals_npz, "adverse_signals_npz")
        _require_bool(self.overwrite, "overwrite")
        if self.policy not in AUDIT_POLICIES:
            raise ValueError(f"policy must be one of {AUDIT_POLICIES}")
        _require_positive_int(self.max_steps, "max_steps")
        _optional_nonnegative_int(self.start_event_index, "start_event_index")
        _require_positive_int(self.cancel_guard_ticks, "cancel_guard_ticks")
        if self.mmap_mode not in (None, "r"):
            raise ValueError('mmap_mode must be None or "r"')
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        _require_positive_float(self.max_order_qty, "max_order_qty")
        _require_positive_int(self.post_only_gap_ticks, "post_only_gap_ticks")
        _require_positive_float(self.default_order_qty, "default_order_qty")
        _require_finite_float(self.action_size_raw, "action_size_raw")
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        _require_probability(self.l2_decrease_weight, "l2_decrease_weight")
        _require_probability(self.trade_at_level_weight, "trade_at_level_weight")
        _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty")
        _require_bool(self.dedupe_l2_decrease_with_trade_prints, "dedupe_l2_decrease_with_trade_prints")
        _require_finite_float(self.maker_fee_bps, "maker_fee_bps")
        _require_finite_float(self.edge_min_executable_edge_bps, "edge_min_executable_edge_bps")
        _require_nonnegative_float(self.edge_latency_buffer_bps, "edge_latency_buffer_bps")
        _require_finite_float(self.edge_inventory_skew_bps_per_unit, "edge_inventory_skew_bps_per_unit")
        _require_nonnegative_int(self.decision_compute_latency_us, "decision_compute_latency_us")
        _require_nonnegative_int(self.order_entry_latency_us, "order_entry_latency_us")
        _require_nonnegative_int(self.cancel_latency_us, "cancel_latency_us")
        _require_nonnegative_float(self.inventory_penalty_bps, "inventory_penalty_bps")
        _require_nonnegative_float(self.turnover_penalty_bps, "turnover_penalty_bps")
        _require_nonnegative_float(self.cancel_penalty, "cancel_penalty")
        _require_nonnegative_float(self.drawdown_penalty_rate, "drawdown_penalty_rate")
        _require_nonnegative_float(self.terminal_inventory_penalty_bps, "terminal_inventory_penalty_bps")
        _require_positive_float(self.reward_scale, "reward_scale")
        _require_bool(self.horizon_diagnostics_enabled, "horizon_diagnostics_enabled")
        object.__setattr__(
            self,
            "horizon_diagnostics_us",
            parse_horizon_diagnostics_us(tuple(self.horizon_diagnostics_us)),
        )
        object.__setattr__(self, "stdout_mode", validate_stdout_mode(self.stdout_mode))


def _default_linear_signals_npz(tape_root: str) -> Path:
    return Path(tape_root) / LINEAR_SIGNALS_FILENAME



def run_execution_sim_audit(config: ExecutionSimAuditConfig) -> dict[str, object]:
    if not isinstance(config, ExecutionSimAuditConfig):
        raise ValueError("config must be ExecutionSimAuditConfig")

    output_path = Path(config.output_json) if config.output_json is not None else Path(config.tape_root) / "audit_execution_sim_summary.json"
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(str(output_path))

    tape = load_execution_tape(
        config.tape_root,
        mmap_mode=config.mmap_mode,
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    linear_signals_path = (
        Path(config.linear_signals_npz)
        if config.linear_signals_npz is not None
        else _default_linear_signals_npz(config.tape_root)
    )
    linear_signals = load_linear_signal_artifact_npz(linear_signals_path)
    decision_grid = load_decision_grid(config.decision_grid_path)
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    adverse_signals = load_adverse_selection_signals(config.adverse_signals_npz) if config.adverse_signals_npz is not None else None
    decision_grid_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_grid=decision_grid,
        requested_start_event_index=config.start_event_index,
        min_rows=(config.max_steps + 1) if config.max_steps is not None else None,
    )
    env_config = build_execution_env_config_from_attrs(
        config,
        adverse_signals_enabled=config.adverse_signals_npz is not None,
    )

    env = ExecutionEnv(tape, config=env_config, decision_grid=decision_grid, linear_signals=linear_signals, adverse_signals=adverse_signals)
    env.reset(start_event_index=config.start_event_index)
    horizon_config = HorizonDiagnosticsConfig(
        enabled=config.horizon_diagnostics_enabled,
        horizons_us=config.horizon_diagnostics_us,
    )
    horizon_accumulator = (
        HorizonDiagnosticsAccumulator.from_execution(
            decision_grid=decision_grid,
            tape=tape,
            linear_signals=linear_signals,
            config=horizon_config,
        )
        if horizon_config.enabled
        else None
    )
    if horizon_accumulator is not None:
        horizon_accumulator.start_episode()

    acc = ExecutionMetricAccumulator()
    while True:
        action = _policy_action(config.policy, acc.step_count, action_size_raw=config.action_size_raw)
        step = env.step(action)
        acc.update(step.execution)
        if horizon_accumulator is not None and horizon_accumulator.enabled:
            horizon_accumulator.record_step(
                step,
                requested_bid_enabled=action.bid_enabled,
                requested_ask_enabled=action.ask_enabled,
            )
        if step.done or step.truncated:
            break

    metrics = acc.as_dict()
    report = diagnose_execution_metrics(metrics, config=ExecutionDiagnosticsConfig())
    queue_metrics = metrics.get("queue", {})
    fill_metrics = metrics.get("fills", {})
    turnover_metrics = metrics.get("turnover", {})
    reward_metrics = metrics.get("reward", {})
    output_path_str = str(output_path)
    horizon_payload = (
        horizon_accumulator.as_dict(include_records=False)
        if horizon_accumulator is not None
        else {
            "enabled": False,
            "horizons_us": list(config.horizon_diagnostics_us),
            "decision_level": {},
            "fill_markouts": {},
            "signal_alignment": {},
            "warnings": [],
        }
    )
    if config.horizon_debug_json is not None and horizon_accumulator is not None:
        write_json_atomic(
            config.horizon_debug_json,
            horizon_accumulator.as_dict(include_records=True),
        )
    debug_output_path = Path(config.debug_output_json) if config.debug_output_json is not None else None
    linear_summary = linear_signal_artifact_summary(linear_signals, path=str(linear_signals_path))
    summary = {
        "status": report.status,
        "run_type": "audit_execution_sim",
        "audit_type": "execution_sim",
        "compact_summary": {},
        "horizon_diagnostics": horizon_payload,
        "tape_root": str(Path(config.tape_root)),
        "decision_grid_path": str(Path(config.decision_grid_path)),
        "output_json": output_path_str,
        "debug_output_json": None if debug_output_path is None else str(debug_output_path),
        "horizon_debug_json": config.horizon_debug_json,
        "config": _summary_config(config),
        "tape": {
            "schema": tape.manifest.schema,
            "exchange": tape.manifest.exchange,
            "symbol": tape.manifest.symbol,
            "num_events": tape.manifest.num_events,
            "num_l2_batches": tape.manifest.num_l2_batches,
            "num_trades": tape.manifest.num_trades,
            "start_local_ts_us": tape.manifest.start_local_ts_us,
            "end_local_ts_us": tape.manifest.end_local_ts_us,
            "book_depth": tape.manifest.notes.get("book_depth") if tape.manifest.notes is not None else None,
        },
        "metrics": metrics,
        "queue_mode_comparison": {
            "queue_mode": config.queue_mode.value,
            "l2_effective_decrease_qty_total": queue_metrics.get("l2_effective_decrease_qty_total"),
            "l2_trade_dedupe_qty_total": queue_metrics.get("l2_trade_dedupe_qty_total"),
            "queue_depletion_fill_count": queue_metrics.get("queue_depletion_fill_count"),
            "fill_reason_counts": fill_metrics.get("reason_counts"),
            "turnover": turnover_metrics,
            "reward": reward_metrics,
            "fill_rate": fill_metrics.get("fill_rate"),
        },
        "diagnostics": report.as_dict(),
        "linear_signals": {
            "schema": linear_summary.get("schema"),
            "path": linear_summary.get("path"),
            "n_rows": linear_summary.get("n_rows"),
            "dtype": linear_summary.get("dtype"),
            "fields": linear_summary.get("fields"),
            "first_decision_event_index": linear_summary.get("first_decision_event_index"),
            "last_decision_event_index": linear_summary.get("last_decision_event_index"),
            "first_decision_local_ts_us": linear_summary.get("first_decision_local_ts_us"),
            "last_decision_local_ts_us": linear_summary.get("last_decision_local_ts_us"),
        },
        "decision_grid_start": decision_grid_start.as_dict(),
        "decision_grid": {
            "schema": decision_grid.metadata.schema,
            "hash": decision_grid.decision_grid_hash,
            "n_rows": decision_grid.n_rows,
            "schedule": decision_grid.decision_schedule,
        },
        "lineage": {
            "decision_grid": {
                "schema": decision_grid.metadata.schema,
                "hash": decision_grid.decision_grid_hash,
                "n_rows": decision_grid.n_rows,
                "schedule": decision_grid.decision_schedule,
            },
            "linear_signals": {
                "schema": linear_summary.get("schema"),
                "path": linear_summary.get("path"),
                "n_rows": linear_summary.get("n_rows"),
                "metadata": linear_summary.get("metadata"),
            },
        },
        "debug": {
            "debug_output_json": None if debug_output_path is None else str(debug_output_path),
            "horizon_debug_json": config.horizon_debug_json,
        },
    }

    summary["compact_summary"] = compact_audit_summary(summary)
    if debug_output_path is not None:
        write_json_atomic(
            debug_output_path,
            {
                "status": report.status,
                "run_type": "audit_execution_sim_debug",
                "primary_output_json": output_path_str,
                "config": _summary_config(config),
                "metrics": metrics,
                "diagnostics": report.as_dict(),
                "linear_signals": linear_summary,
            },
        )
    write_json_atomic(output_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an execution tape with a deterministic simple quote policy.")
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--decision-grid", dest="decision_grid_path", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--debug-output-json")
    parser.add_argument("--horizon-debug-json")
    parser.add_argument(
        "--linear-signals-npz",
        help="Canonical no-move-gated linear signal NPZ. Defaults to <tape-root>/linear_signals.npz. Required; missing file is an error.",
    )
    parser.add_argument("--adverse-signals-npz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--policy", choices=AUDIT_POLICIES, default="alternate_bid_ask")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--cancel-guard-ticks", type=int, default=2)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--max-distance-ticks", type=int, default=1)
    parser.add_argument("--max-order-qty", type=float, default=0.001)
    parser.add_argument("--post-only-gap-ticks", type=int, default=1)
    parser.add_argument("--default-order-qty", type=float, default=0.001)
    parser.add_argument("--action-size-raw", type=float, default=100.0)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"), default="conservative")
    parser.add_argument("--l2-decrease-weight", type=float, default=0.25)
    parser.add_argument("--trade-at-level-weight", type=float, default=0.5)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float, default=1000000000.0)
    parser.add_argument(
        "--no-dedupe-l2-decrease-with-trade-prints",
        action="store_true",
        help="Disable de-duplication of L2 visible decreases already explained by same-level trade prints.",
    )
    parser.add_argument("--maker-fee-bps", type=float, default=-0.5)
    parser.add_argument("--edge-min-executable-edge-bps", type=float, default=0.0)
    parser.add_argument("--edge-latency-buffer-bps", type=float, default=0.0)
    parser.add_argument("--edge-inventory-skew-bps-per-unit", type=float, default=0.0)
    parser.add_argument("--decision-compute-latency-us", type=int, default=50)
    parser.add_argument("--order-entry-latency-us", type=int, default=500)
    parser.add_argument("--cancel-latency-us", type=int, default=500)
    parser.add_argument("--inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument("--cancel-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty-rate", type=float, default=0.0)
    parser.add_argument("--terminal-inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--horizon-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable compact future-horizon reward/markout diagnostics.",
    )
    parser.add_argument(
        "--horizon-diagnostics-us",
        default="250000,500000,1000000",
        help="Comma-separated positive future horizons in microseconds.",
    )
    parser.add_argument("--stdout-mode", choices=STDOUT_MODES, default="summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExecutionSimAuditConfig(
        tape_root=args.tape_root,
        decision_grid_path=args.decision_grid_path,
        output_json=args.output_json,
        debug_output_json=args.debug_output_json,
        horizon_debug_json=args.horizon_debug_json,
        linear_signals_npz=args.linear_signals_npz,
        adverse_signals_npz=args.adverse_signals_npz,
        overwrite=args.overwrite,
        policy=args.policy,
        max_steps=args.max_steps,
        start_event_index=args.start_event_index,
        cancel_guard_ticks=args.cancel_guard_ticks,
        mmap_mode=None if args.no_mmap else "r",
        max_distance_ticks=args.max_distance_ticks,
        max_order_qty=args.max_order_qty,
        post_only_gap_ticks=args.post_only_gap_ticks,
        default_order_qty=args.default_order_qty,
        action_size_raw=args.action_size_raw,
        queue_mode=args.queue_mode,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
        dedupe_l2_decrease_with_trade_prints=not args.no_dedupe_l2_decrease_with_trade_prints,
        maker_fee_bps=args.maker_fee_bps,
        edge_min_executable_edge_bps=args.edge_min_executable_edge_bps,
        edge_latency_buffer_bps=args.edge_latency_buffer_bps,
        edge_inventory_skew_bps_per_unit=args.edge_inventory_skew_bps_per_unit,
        decision_compute_latency_us=args.decision_compute_latency_us,
        order_entry_latency_us=args.order_entry_latency_us,
        cancel_latency_us=args.cancel_latency_us,
        inventory_penalty_bps=args.inventory_penalty_bps,
        turnover_penalty_bps=args.turnover_penalty_bps,
        cancel_penalty=args.cancel_penalty,
        drawdown_penalty_rate=args.drawdown_penalty_rate,
        terminal_inventory_penalty_bps=args.terminal_inventory_penalty_bps,
        reward_scale=args.reward_scale,
        horizon_diagnostics_enabled=args.horizon_diagnostics,
        horizon_diagnostics_us=parse_horizon_diagnostics_us(args.horizon_diagnostics_us),
        stdout_mode=args.stdout_mode,
    )
    summary = run_execution_sim_audit(config)
    if config.stdout_mode == "summary":
        print_human_summary("audit_execution_sim", summary)
    elif config.stdout_mode == "json":
        print(compact_json_line(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
