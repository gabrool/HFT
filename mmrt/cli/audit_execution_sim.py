"""Audit an existing execution tape by running a deterministic execution simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from mmrt.execution.contracts import ActionSpec, PositionState, QueueModelMode
from mmrt.execution.diagnostics import ExecutionDiagnosticsConfig, diagnose_execution_metrics
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.execution_tape import load_execution_tape
from mmrt.execution.fill_sim import FillSimulatorConfig
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    load_linear_signal_arrays_npz,
    linear_signal_arrays_summary,
)
from mmrt.execution.metrics import ExecutionMetricAccumulator
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.quote_geometry import ContinuousQuoteAction, QuoteGeometryConfig
from mmrt.execution.reward import RewardConfig

AUDIT_POLICIES = (
    "disabled",
    "bid",
    "ask",
    "two_sided",
    "alternate_bid_ask",
)


def _policy_action(policy: str, step_index: int, *, action_size_raw: float) -> ContinuousQuoteAction:
    if policy not in AUDIT_POLICIES:
        raise ValueError(f"policy must be one of {AUDIT_POLICIES}")
    if policy == "disabled":
        bid_logit, ask_logit = -1.0, -1.0
    elif policy == "bid":
        bid_logit, ask_logit = 1.0, -1.0
    elif policy == "ask":
        bid_logit, ask_logit = -1.0, 1.0
    elif policy == "two_sided":
        bid_logit, ask_logit = 1.0, 1.0
    else:
        bid_logit, ask_logit = (1.0, -1.0) if step_index % 2 == 0 else (-1.0, 1.0)
    return ContinuousQuoteAction(
        bid_enable_logit=bid_logit,
        ask_enable_logit=ask_logit,
        bid_distance_raw=0.0,
        ask_distance_raw=0.0,
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


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


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
        "linear_signals_npz": config.linear_signals_npz,
        "policy": config.policy,
        "max_steps": config.max_steps,
        "start_event_index": config.start_event_index,
        "decision_interval_us": config.decision_interval_us,
        "mmap_mode": config.mmap_mode,
        "max_distance_ticks": config.max_distance_ticks,
        "max_order_qty": config.max_order_qty,
        "min_distance_ticks": config.min_distance_ticks,
        "default_order_qty": config.default_order_qty,
        "action_size_raw": config.action_size_raw,
        "queue_mode": config.queue_mode.value,
        "l2_decrease_weight": config.l2_decrease_weight,
        "trade_at_level_weight": config.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.unknown_level_queue_ahead_qty,
        "maker_fee_bps": config.maker_fee_bps,
        "inventory_penalty_bps": config.inventory_penalty_bps,
        "turnover_penalty_bps": config.turnover_penalty_bps,
        "cancel_penalty": config.cancel_penalty,
        "drawdown_penalty_rate": config.drawdown_penalty_rate,
        "terminal_inventory_penalty_bps": config.terminal_inventory_penalty_bps,
    }


@dataclass(frozen=True, slots=True)
class ExecutionSimAuditConfig:
    tape_root: str
    output_json: str | None = None
    linear_signals_npz: str | None = None
    overwrite: bool = False

    policy: str = "alternate_bid_ask"
    max_steps: int = 1_000
    start_event_index: int | None = None
    decision_interval_us: int = 500_000
    mmap_mode: str | None = "r"

    max_distance_ticks: int = 1
    max_order_qty: float = 0.001
    min_distance_ticks: int = 1
    default_order_qty: float = 0.001
    action_size_raw: float = 100.0

    queue_mode: QueueModelMode | str = QueueModelMode.BALANCED
    l2_decrease_weight: float = 1.0
    trade_at_level_weight: float = 1.0
    unknown_level_queue_ahead_qty: float = 0.0

    maker_fee_bps: float = 0.0

    inventory_penalty_bps: float = 0.0
    turnover_penalty_bps: float = 0.0
    cancel_penalty: float = 0.0
    drawdown_penalty_rate: float = 0.0
    terminal_inventory_penalty_bps: float = 0.0

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        if self.output_json is not None:
            _require_nonempty_str(self.output_json, "output_json")
        if self.linear_signals_npz is not None:
            _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        _require_bool(self.overwrite, "overwrite")
        if self.policy not in AUDIT_POLICIES:
            raise ValueError(f"policy must be one of {AUDIT_POLICIES}")
        _require_positive_int(self.max_steps, "max_steps")
        _optional_nonnegative_int(self.start_event_index, "start_event_index")
        _require_positive_int(self.decision_interval_us, "decision_interval_us")
        if self.mmap_mode not in (None, "r"):
            raise ValueError('mmap_mode must be None or "r"')
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        _require_positive_float(self.max_order_qty, "max_order_qty")
        _require_positive_int(self.min_distance_ticks, "min_distance_ticks")
        _require_positive_float(self.default_order_qty, "default_order_qty")
        _require_finite_float(self.action_size_raw, "action_size_raw")
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        _require_probability(self.l2_decrease_weight, "l2_decrease_weight")
        _require_probability(self.trade_at_level_weight, "trade_at_level_weight")
        _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty")
        _require_nonnegative_float(self.maker_fee_bps, "maker_fee_bps")
        _require_nonnegative_float(self.inventory_penalty_bps, "inventory_penalty_bps")
        _require_nonnegative_float(self.turnover_penalty_bps, "turnover_penalty_bps")
        _require_nonnegative_float(self.cancel_penalty, "cancel_penalty")
        _require_nonnegative_float(self.drawdown_penalty_rate, "drawdown_penalty_rate")
        _require_nonnegative_float(self.terminal_inventory_penalty_bps, "terminal_inventory_penalty_bps")


def _default_linear_signals_npz(tape_root: str) -> Path:
    return Path(tape_root) / LINEAR_SIGNALS_FILENAME


def run_execution_sim_audit(config: ExecutionSimAuditConfig) -> dict[str, object]:
    if not isinstance(config, ExecutionSimAuditConfig):
        raise ValueError("config must be ExecutionSimAuditConfig")

    output_path = Path(config.output_json) if config.output_json is not None else Path(config.tape_root) / "audit_execution_sim_summary.json"
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(str(output_path))

    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode)
    linear_signals_path = (
        Path(config.linear_signals_npz)
        if config.linear_signals_npz is not None
        else _default_linear_signals_npz(config.tape_root)
    )
    linear_signals = load_linear_signal_arrays_npz(linear_signals_path)
    env_config = ExecutionEnvConfig(
        decision_interval_us=config.decision_interval_us,
        action_spec=ActionSpec(
            max_distance_ticks=config.max_distance_ticks,
            max_order_qty=config.max_order_qty,
        ),
        quote_geometry_config=QuoteGeometryConfig(
            min_distance_ticks=config.min_distance_ticks,
            default_order_qty=config.default_order_qty,
        ),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=config.queue_mode,
                l2_decrease_weight=config.l2_decrease_weight,
                trade_at_level_weight=config.trade_at_level_weight,
                unknown_level_queue_ahead_qty=config.unknown_level_queue_ahead_qty,
            ),
            maker_fee_bps=config.maker_fee_bps,
        ),
        reward_config=RewardConfig(
            inventory_penalty_bps=config.inventory_penalty_bps,
            turnover_penalty_bps=config.turnover_penalty_bps,
            cancel_penalty=config.cancel_penalty,
            drawdown_penalty_rate=config.drawdown_penalty_rate,
            terminal_inventory_penalty_bps=config.terminal_inventory_penalty_bps,
        ),
        initial_position=PositionState(),
        max_episode_steps=config.max_steps,
    )

    env = ExecutionEnv(tape, config=env_config, linear_signals=linear_signals)
    env.reset(start_event_index=config.start_event_index)

    acc = ExecutionMetricAccumulator()
    while True:
        action = _policy_action(config.policy, acc.step_count, action_size_raw=config.action_size_raw)
        step = env.step(action)
        acc.update(step.execution)
        if step.done or step.truncated:
            break

    metrics = acc.as_dict()
    report = diagnose_execution_metrics(metrics, config=ExecutionDiagnosticsConfig())
    output_path_str = str(output_path)
    summary = {
        "status": report.status,
        "audit_type": "execution_sim",
        "tape_root": str(Path(config.tape_root)),
        "output_json": output_path_str,
        "config": _summary_config(config),
        "tape": {
            "schema_version": tape.manifest.schema_version,
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
        "diagnostics": report.as_dict(),
        "linear_signals": linear_signal_arrays_summary(linear_signals, path=str(linear_signals_path)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an execution tape with a deterministic simple quote policy.")
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--linear-signals-npz",
        help="Canonical no-move-gated linear signal NPZ. Defaults to <tape-root>/linear_signals.npz. Required; missing file is an error.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--policy", choices=AUDIT_POLICIES, default="alternate_bid_ask")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--decision-interval-us", type=int, default=500000)
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--max-distance-ticks", type=int, default=1)
    parser.add_argument("--max-order-qty", type=float, default=0.001)
    parser.add_argument("--min-distance-ticks", type=int, default=1)
    parser.add_argument("--default-order-qty", type=float, default=0.001)
    parser.add_argument("--action-size-raw", type=float, default=100.0)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"), default="balanced")
    parser.add_argument("--l2-decrease-weight", type=float, default=1.0)
    parser.add_argument("--trade-at-level-weight", type=float, default=1.0)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float, default=0.0)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument("--cancel-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty-rate", type=float, default=0.0)
    parser.add_argument("--terminal-inventory-penalty-bps", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExecutionSimAuditConfig(
        tape_root=args.tape_root,
        output_json=args.output_json,
        linear_signals_npz=args.linear_signals_npz,
        overwrite=args.overwrite,
        policy=args.policy,
        max_steps=args.max_steps,
        start_event_index=args.start_event_index,
        decision_interval_us=args.decision_interval_us,
        mmap_mode=None if args.no_mmap else "r",
        max_distance_ticks=args.max_distance_ticks,
        max_order_qty=args.max_order_qty,
        min_distance_ticks=args.min_distance_ticks,
        default_order_qty=args.default_order_qty,
        action_size_raw=args.action_size_raw,
        queue_mode=args.queue_mode,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
        maker_fee_bps=args.maker_fee_bps,
        inventory_penalty_bps=args.inventory_penalty_bps,
        turnover_penalty_bps=args.turnover_penalty_bps,
        cancel_penalty=args.cancel_penalty,
        drawdown_penalty_rate=args.drawdown_penalty_rate,
        terminal_inventory_penalty_bps=args.terminal_inventory_penalty_bps,
    )
    summary = run_execution_sim_audit(config)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
