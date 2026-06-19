"""Shared JSON and human-facing CLI output helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, TextIO

STDOUT_MODES = ("summary", "json", "none")

__all__ = [
    "STDOUT_MODES",
    "compact_audit_summary",
    "compact_eval_summary",
    "compact_json_line",
    "compact_training_summary",
    "print_human_summary",
    "validate_stdout_mode",
    "write_json_atomic",
]


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(target)


def compact_json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_stdout_mode(value: str) -> str:
    if value not in STDOUT_MODES:
        raise ValueError(f"stdout_mode must be one of {STDOUT_MODES}")
    return value


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _get(mapping: Mapping[str, object], path: tuple[str, ...], default: object = None) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key, default)
    return current


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _quote_rates_from_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    steps = _mapping(metrics.get("steps"))
    orders = _mapping(metrics.get("orders"))
    count = int(steps.get("count", 0) or 0)
    denom = max(count, 1)
    bid = int(orders.get("quote_bid_enabled_count", 0) or 0)
    ask = int(orders.get("quote_ask_enabled_count", 0) or 0)
    two = int(orders.get("two_sided_quote_count", 0) or 0)
    none = int(orders.get("all_quotes_disabled_count", 0) or 0)
    return {
        "no_quote": none / denom,
        "bid_only": max(bid - two, 0) / denom,
        "ask_only": max(ask - two, 0) / denom,
        "two_sided": two / denom,
    }


def _quote_rates_from_telemetry(telemetry: Mapping[str, object], fallback_metrics: Mapping[str, object]) -> dict[str, float]:
    effective = _mapping(telemetry.get("effective_quotes"))
    if effective:
        return {
            "no_quote": float(effective.get("quote_no_quote_rate", 0.0) or 0.0),
            "bid_only": float(effective.get("quote_bid_only_rate", 0.0) or 0.0),
            "ask_only": float(effective.get("quote_ask_only_rate", 0.0) or 0.0),
            "two_sided": float(effective.get("quote_two_sided_rate", 0.0) or 0.0),
        }
    return _quote_rates_from_metrics(fallback_metrics)


def _horizon_1s(payload: Mapping[str, object]) -> Mapping[str, object]:
    horizon = payload.get("horizon_diagnostics")
    if not isinstance(horizon, Mapping):
        return {}
    decision = _mapping(horizon.get("decision_level"))
    fills = _mapping(horizon.get("fill_markouts"))
    decision_h = _mapping(_mapping(decision.get("by_horizon")).get("1000000"))
    fill_h = _mapping(_mapping(fills.get("by_horizon")).get("1000000"))
    decision_all = _mapping(decision_h.get("all"))
    fill_all = _mapping(fill_h.get("all"))
    return {
        "actual_path_equity_delta_mean": decision_all.get("actual_path_equity_delta_mean"),
        "carry_mark_equity_delta_mean": decision_all.get("carry_mark_equity_delta_mean"),
        "fill_net_markout_bps_mean": fill_all.get("net_markout_bps_mean"),
    }


def _fill_reason_counts(metrics: Mapping[str, object]) -> Mapping[str, object]:
    fills = _mapping(metrics.get("fills"))
    return _mapping(fills.get("reason_counts"))


def compact_eval_summary(summary: Mapping[str, object]) -> dict[str, object]:
    evaluation = _mapping(summary.get("evaluation"))
    metrics = _mapping(evaluation.get("metrics") or summary.get("metrics"))
    rewards = _mapping(metrics.get("rewards"))
    fills = _mapping(metrics.get("fills"))
    equity = _mapping(metrics.get("equity"))
    position = _mapping(metrics.get("position"))
    telemetry = _mapping(summary.get("policy_action_telemetry") or evaluation.get("telemetry"))
    adverse = _mapping(summary.get("adverse_signal_queue_config"))
    compact = {
        "status": summary.get("status"),
        "run_type": "evaluate_execution_policy",
        "eval_split": summary.get("eval_split"),
        "steps": evaluation.get("steps", _get(metrics, ("steps", "count"))),
        "reward_total": rewards.get("total_raw"),
        "reward_mean": rewards.get("mean"),
        "fills": fills.get("count"),
        "fill_rate": fills.get("fill_rate"),
        "final_equity": equity.get("final"),
        "net_inventory_qty": position.get("final_inventory_qty"),
        "queue_mode": _get(summary, ("env_config", "queue_mode"), _get(summary, ("config", "queue_mode"))),
        "adverse_queue_config_status": adverse.get("status") if adverse else None,
        "effective_quote_rates": _quote_rates_from_telemetry(telemetry, metrics),
        "fill_reason_counts": _fill_reason_counts(metrics),
        "horizon_1s": dict(_horizon_1s(summary)),
        "output_json": summary.get("output_json"),
    }
    checkpoint = _mapping(summary.get("checkpoint"))
    if "updates_completed" in checkpoint:
        compact["checkpoint_updates"] = checkpoint.get("updates_completed")
    deterministic = _get(evaluation, ("config", "deterministic"))
    if deterministic is not None:
        compact["deterministic"] = deterministic
    device = _mapping(summary.get("device"))
    if device:
        compact["device"] = device.get("resolved_device") or device.get("device") or device.get("resolved")
    return compact


def compact_audit_summary(summary: Mapping[str, object]) -> dict[str, object]:
    metrics = _mapping(summary.get("metrics"))
    rewards = _mapping(metrics.get("rewards"))
    fills = _mapping(metrics.get("fills"))
    return {
        "status": summary.get("status"),
        "run_type": "audit_execution_sim",
        "policy": _get(summary, ("config", "policy")),
        "queue_mode": _get(summary, ("config", "queue_mode")),
        "steps": _get(metrics, ("steps", "count")),
        "reward_total": rewards.get("total_raw"),
        "reward_mean": rewards.get("mean"),
        "fills": fills.get("count"),
        "fill_rate": fills.get("fill_rate"),
        "fill_reason_counts": _fill_reason_counts(metrics),
        "horizon_1s": dict(_horizon_1s(summary)),
        "output_json": summary.get("output_json"),
    }


def compact_training_summary(summary: Mapping[str, object]) -> dict[str, object]:
    training = _mapping(summary.get("training"))
    final = _mapping(training.get("final"))
    rollout = _mapping(final.get("rollout"))
    ppo = _mapping(final.get("ppo"))
    reward_projection = _mapping(final.get("reward_projection_stats"))
    telemetry = _mapping(final.get("telemetry_brief") or final.get("telemetry"))
    effective_rates = _mapping(telemetry.get("effective_quote_rates"))
    if not effective_rates:
        effective_rates = _mapping(telemetry.get("quote_mode_rates"))
    if not effective_rates and any(key in telemetry for key in ("quote_no_quote_rate", "quote_bid_only_rate", "quote_ask_only_rate", "quote_two_sided_rate")):
        effective_rates = {
            "no_quote": telemetry.get("quote_no_quote_rate"),
            "bid_only": telemetry.get("quote_bid_only_rate"),
            "ask_only": telemetry.get("quote_ask_only_rate"),
            "two_sided": telemetry.get("quote_two_sided_rate"),
        }
    discounting = _mapping(rollout.get("discounting"))
    config = _mapping(summary.get("config"))
    compact = {
        "status": summary.get("status"),
        "run_type": summary.get("run_type", "train_execution_ppo"),
        "updates": training.get("updates_completed"),
        "device": _get(summary, ("device", "resolved_device")),
        "queue_mode": _get(summary, ("env_config", "queue_mode")),
        "adverse_queue_config_status": _get(summary, ("adverse_signal_queue_config", "status")),
        "discount_mode": discounting.get("discount_mode", config.get("discount_mode")),
        "discount_horizon_us": discounting.get("discount_horizon_us", config.get("discount_horizon_us")),
        "training_reward_mode": config.get(
            "training_reward_mode",
            _get(training, ("config", "rollout_config", "reward_config", "training_reward_mode")),
        ),
        "reward_valid_fraction": reward_projection.get("valid_fraction"),
        "projected_reward_mean": reward_projection.get("projected_reward_mean"),
        "env_reward_mean": reward_projection.get("env_reward_mean"),
        "gamma": _get(training, ("config", "rollout_config", "gamma"), config.get("gamma")),
        "gae_lambda": _get(training, ("config", "rollout_config", "gae_lambda"), config.get("gae_lambda")),
        "final_reward_mean": rollout.get("reward_mean"),
        "final_return_mean": rollout.get("return_mean"),
        "final_entropy": ppo.get("entropy"),
        "approx_kl": ppo.get("approx_kl"),
        "effective_quote_rates": dict(effective_rates),
        "checkpoint": summary.get("checkpoint_path"),
        "output_json": summary.get("output_json"),
    }
    return compact


def _print_rates(prefix: str, rates: Mapping[str, object], stream: TextIO) -> None:
    if not rates:
        return
    print(
        f"{prefix}: no_quote={_fmt(rates.get('no_quote'))} "
        f"bid_only={_fmt(rates.get('bid_only'))} "
        f"ask_only={_fmt(rates.get('ask_only'))} "
        f"two_sided={_fmt(rates.get('two_sided'))}",
        file=stream,
    )


def _print_fills(reason_counts: Mapping[str, object], stream: TextIO) -> None:
    if not reason_counts:
        return
    print(
        "fills: "
        f"trade_through={_fmt(reason_counts.get('trade_through'))} "
        f"trade_at_level={_fmt(reason_counts.get('trade_at_level'))} "
        f"queue_depletion={_fmt(reason_counts.get('queue_depletion'))}",
        file=stream,
    )


def _print_horizon(horizon: Mapping[str, object], stream: TextIO) -> None:
    if not horizon:
        return
    print(
        "horizon_1s: "
        f"path_mean={_fmt(horizon.get('actual_path_equity_delta_mean'))} "
        f"carry_mean={_fmt(horizon.get('carry_mark_equity_delta_mean'))} "
        f"fill_net_markout_bps={_fmt(horizon.get('fill_net_markout_bps_mean'))}",
        file=stream,
    )


def print_human_summary(kind: str, payload: Mapping[str, object], *, stream: TextIO | None = None) -> None:
    out = sys.stdout if stream is None else stream
    if kind == "evaluate_execution_policy":
        summary = compact_eval_summary(payload)
        print(f"evaluate_execution_policy: {_fmt(summary.get('status'))}", file=out)
        print(
            f"split={_fmt(summary.get('eval_split'))} steps={_fmt(summary.get('steps'))} "
            f"deterministic={_fmt(summary.get('deterministic'))} device={_fmt(summary.get('device'))}",
            file=out,
        )
        print(
            f"queue={_fmt(summary.get('queue_mode'))} "
            f"adverse_config={_fmt(summary.get('adverse_queue_config_status'))} "
            f"checkpoint_updates={_fmt(summary.get('checkpoint_updates'))}",
            file=out,
        )
        print(
            f"reward_total={_fmt(summary.get('reward_total'))} "
            f"reward_mean={_fmt(summary.get('reward_mean'))} "
            f"fills={_fmt(summary.get('fills'))} fill_rate={_fmt(summary.get('fill_rate'))}",
            file=out,
        )
        _print_rates("modes", _mapping(summary.get("effective_quote_rates")), out)
        _print_fills(_mapping(summary.get("fill_reason_counts")), out)
        _print_horizon(_mapping(summary.get("horizon_1s")), out)
        print(f"output_json={_fmt(summary.get('output_json'))}", file=out)
        return

    if kind == "audit_execution_sim":
        summary = compact_audit_summary(payload)
        print(f"audit_execution_sim: {_fmt(summary.get('status'))}", file=out)
        print(
            f"policy={_fmt(summary.get('policy'))} queue={_fmt(summary.get('queue_mode'))} "
            f"steps={_fmt(summary.get('steps'))}",
            file=out,
        )
        print(
            f"reward_total={_fmt(summary.get('reward_total'))} "
            f"reward_mean={_fmt(summary.get('reward_mean'))} "
            f"fills={_fmt(summary.get('fills'))} fill_rate={_fmt(summary.get('fill_rate'))}",
            file=out,
        )
        _print_fills(_mapping(summary.get("fill_reason_counts")), out)
        _print_horizon(_mapping(summary.get("horizon_1s")), out)
        print(f"output_json={_fmt(summary.get('output_json'))}", file=out)
        return

    if kind in ("train_execution_ppo", "profile_execution_ppo_rollout"):
        summary = compact_training_summary(payload)
        print(f"{kind}: {_fmt(summary.get('status'))}", file=out)
        print(
            f"updates={_fmt(summary.get('updates'))} device={_fmt(summary.get('device'))} "
            f"queue={_fmt(summary.get('queue_mode'))} "
            f"adverse_config={_fmt(summary.get('adverse_queue_config_status'))}",
            file=out,
        )
        print(
            f"discount={_fmt(summary.get('discount_mode'))} "
            f"horizon_us={_fmt(summary.get('discount_horizon_us'))} "
            f"gamma={_fmt(summary.get('gamma'))} lambda={_fmt(summary.get('gae_lambda'))}",
            file=out,
        )
        print(
            f"final_reward_mean={_fmt(summary.get('final_reward_mean'))} "
            f"final_return_mean={_fmt(summary.get('final_return_mean'))} "
            f"final_entropy={_fmt(summary.get('final_entropy'))} "
            f"approx_kl={_fmt(summary.get('approx_kl'))}",
            file=out,
        )
        print(
            f"reward_mode={_fmt(summary.get('training_reward_mode'))} "
            f"valid={_fmt(summary.get('reward_valid_fraction'))} "
            f"projected_reward_mean={_fmt(summary.get('projected_reward_mean'))} "
            f"env_reward_mean={_fmt(summary.get('env_reward_mean'))}",
            file=out,
        )
        _print_rates("modes_final", _mapping(summary.get("effective_quote_rates")), out)
        print(f"checkpoint={_fmt(summary.get('checkpoint'))}", file=out)
        print(f"output_json={_fmt(summary.get('output_json'))}", file=out)
        return

    print(compact_json_line(payload), file=out)
