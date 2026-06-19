"""Compact aggregate telemetry for execution PPO policy actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mmrt.execution.contracts import (
    ExecutionStepResult,
    FillReason,
    OrderSide,
)


ACTION_TELEMETRY_SCHEMA = "mmrt_execution_policy_action_telemetry_v1"

NO_QUOTE = "no_quote"
BID_ONLY = "bid_only"
ASK_ONLY = "ask_only"
TWO_SIDED = "two_sided"
QUOTE_MODE_NAMES = (NO_QUOTE, BID_ONLY, ASK_ONLY, TWO_SIDED)

ENABLE_COMPONENT_NAMES = (
    "bid_enabled",
    "ask_enabled",
    "bid_cancel_guard",
    "ask_cancel_guard",
)
CONTINUOUS_COMPONENT_NAMES = (
    "bid_price_raw",
    "ask_price_raw",
    "bid_size_raw",
    "ask_size_raw",
)


class QuoteMode(str, Enum):
    NO_QUOTE = NO_QUOTE
    BID_ONLY = BID_ONLY
    ASK_ONLY = ASK_ONLY
    TWO_SIDED = TWO_SIDED


_MODE_TO_ID = {name: idx for idx, name in enumerate(QUOTE_MODE_NAMES)}


def quote_mode_from_bools(bid_enabled: bool, ask_enabled: bool) -> str:
    bid = bool(bid_enabled)
    ask = bool(ask_enabled)
    if bid and ask:
        return TWO_SIDED
    if bid:
        return BID_ONLY
    if ask:
        return ASK_ONLY
    return NO_QUOTE


def quote_mode_id_from_bools(bid_enabled: bool, ask_enabled: bool) -> int:
    return _MODE_TO_ID[quote_mode_from_bools(bid_enabled, ask_enabled)]


def quote_mode_from_id(mode_id: int) -> str:
    if mode_id < 0 or mode_id >= len(QUOTE_MODE_NAMES):
        raise ValueError("mode_id is outside quote mode range")
    return QUOTE_MODE_NAMES[int(mode_id)]


def _json_float(value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("telemetry value must be finite")
    return out


def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _bool_value(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_float_array(values: object, *, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().to("cpu").numpy()
    else:
        array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        array = array.astype(np.float64)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return np.asarray(array, dtype=np.float64)


def _as_2d_float_array(values: object, *, width: int, name: str) -> np.ndarray:
    array = _as_float_array(values, name=name)
    if array.ndim == 1:
        if array.shape[0] != width:
            raise ValueError(f"{name} must have {width} columns")
        array = array.reshape(1, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _flatten_finite(values: object, *, name: str) -> np.ndarray:
    array = _as_float_array(values, name=name).reshape(-1)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _quantile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])
    weight = position - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _empty_scalar_stats() -> dict[str, object]:
    return {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "p05": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


def scalar_stats(values: object) -> dict[str, object]:
    array = _flatten_finite(values, name="values")
    if array.size == 0:
        return _empty_scalar_stats()
    sorted_values = sorted(float(value) for value in array)
    mean = float(array.mean())
    return {
        "count": int(array.size),
        "mean": mean,
        "std": float(array.std(ddof=0)),
        "min": float(sorted_values[0]),
        "p05": _quantile(sorted_values, 0.05),
        "p50": _quantile(sorted_values, 0.50),
        "p95": _quantile(sorted_values, 0.95),
        "max": float(sorted_values[-1]),
    }


@dataclass(slots=True)
class _ScalarAccumulator:
    max_quantile_samples: int = 8192
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    _samples: list[float] = field(default_factory=list)

    def update(self, value: object) -> None:
        finite = _finite_float_or_none(value)
        if finite is None:
            return
        self.count += 1
        self.total += finite
        self.total_sq += finite * finite
        self.min_value = finite if self.min_value is None else min(self.min_value, finite)
        self.max_value = finite if self.max_value is None else max(self.max_value, finite)
        if len(self._samples) < self.max_quantile_samples:
            self._samples.append(finite)
            return
        slot = ((self.count * 1103515245 + 12345) & 0x7FFFFFFF) % self.count
        if slot < self.max_quantile_samples:
            self._samples[int(slot)] = finite

    def update_many(self, values: object, *, name: str = "values") -> None:
        for value in _flatten_finite(values, name=name):
            self.update(float(value))

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.total / self.count
        var = max(0.0, self.total_sq / self.count - mean * mean)
        return math.sqrt(var)

    def as_dict(self) -> dict[str, object]:
        if self.count == 0:
            return _empty_scalar_stats()
        samples = sorted(self._samples)
        return {
            "count": int(self.count),
            "mean": _json_float(self.mean or 0.0),
            "std": _json_float(self.std or 0.0),
            "min": _json_float(self.min_value if self.min_value is not None else 0.0),
            "p05": _quantile(samples, 0.05),
            "p50": _quantile(samples, 0.50),
            "p95": _quantile(samples, 0.95),
            "max": _json_float(self.max_value if self.max_value is not None else 0.0),
        }


@dataclass(slots=True)
class _OutcomeByModeAccumulator:
    max_quantile_samples: int = 8192
    step_count: int = 0
    reward_sum: float = 0.0
    raw_equity_delta_sum: float = 0.0
    fee_total: float = 0.0
    turnover_notional: float = 0.0
    fill_count: int = 0
    fill_step_count: int = 0
    buy_fill_count: int = 0
    sell_fill_count: int = 0
    net_qty: float = 0.0
    fill_reason_counts: dict[str, int] = field(
        default_factory=lambda: {reason.value: 0 for reason in FillReason}
    )
    cancel_count: int = 0
    post_only_reject_count: int = 0
    reward_stat: _ScalarAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.reward_stat = _ScalarAccumulator(self.max_quantile_samples)

    def update(self, step: ExecutionStepResult) -> None:
        if not isinstance(step, ExecutionStepResult):
            raise TypeError("step must be ExecutionStepResult")
        info = step.info or {}
        reward = step.reward
        reward_value = reward.total_reward
        self.step_count += 1
        self.reward_sum += reward_value
        self.reward_stat.update(reward_value)
        self.raw_equity_delta_sum += reward.raw_equity_delta
        self.turnover_notional += _finite_float_or_none(info.get("turnover_notional")) or 0.0
        self.cancel_count += _int_value(info.get("cancel_count"), 0)
        self.post_only_reject_count += _int_value(info.get("post_only_reject_count"), 0)

        if step.fills:
            self.fill_step_count += 1
        for fill in step.fills:
            self.fill_count += 1
            self.fee_total += fill.fee
            self.fill_reason_counts[fill.reason.value] = (
                self.fill_reason_counts.get(fill.reason.value, 0) + 1
            )
            if fill.side == OrderSide.BUY:
                self.buy_fill_count += 1
                self.net_qty += fill.qty
            elif fill.side == OrderSide.SELL:
                self.sell_fill_count += 1
                self.net_qty -= fill.qty

    def as_dict(self, *, total_steps: int) -> dict[str, object]:
        denominator = max(self.step_count, 1)
        reward_stats = self.reward_stat.as_dict()
        return {
            "step_count": int(self.step_count),
            "step_fraction": float(self.step_count / total_steps) if total_steps > 0 else 0.0,
            "reward_sum": float(self.reward_sum),
            "reward_mean": reward_stats["mean"],
            "reward_std": reward_stats["std"],
            "raw_equity_delta_sum": float(self.raw_equity_delta_sum),
            "fee_total": float(self.fee_total),
            "turnover_notional": float(self.turnover_notional),
            "fill_count": int(self.fill_count),
            "fill_step_count": int(self.fill_step_count),
            "fill_step_rate": float(self.fill_step_count / denominator),
            "buy_fill_count": int(self.buy_fill_count),
            "sell_fill_count": int(self.sell_fill_count),
            "net_qty": float(self.net_qty),
            "fill_reason_counts": dict(self.fill_reason_counts),
            "cancel_count": int(self.cancel_count),
            "post_only_reject_count": int(self.post_only_reject_count),
        }


@dataclass(slots=True)
class _TrainingByModeAccumulator:
    max_quantile_samples: int = 8192
    advantage: _ScalarAccumulator = field(init=False)
    returns: _ScalarAccumulator = field(init=False)
    value: _ScalarAccumulator = field(init=False)
    reward: _ScalarAccumulator = field(init=False)

    def __post_init__(self) -> None:
        self.advantage = _ScalarAccumulator(self.max_quantile_samples)
        self.returns = _ScalarAccumulator(self.max_quantile_samples)
        self.value = _ScalarAccumulator(self.max_quantile_samples)
        self.reward = _ScalarAccumulator(self.max_quantile_samples)

    @property
    def count(self) -> int:
        return self.advantage.count

    def update_many(
        self,
        *,
        advantages: object,
        returns: object,
        values: object,
        rewards: object,
    ) -> None:
        self.advantage.update_many(advantages, name="advantages")
        self.returns.update_many(returns, name="returns")
        self.value.update_many(values, name="values")
        self.reward.update_many(rewards, name="rewards")

    def as_dict(self) -> dict[str, object]:
        advantage = self.advantage.as_dict()
        returns = self.returns.as_dict()
        value = self.value.as_dict()
        reward = self.reward.as_dict()
        return {
            "count": int(self.count),
            "advantage_mean": advantage["mean"],
            "advantage_std": advantage["std"],
            "return_mean": returns["mean"],
            "return_std": returns["std"],
            "value_mean": value["mean"],
            "value_std": value["std"],
            "reward_mean": reward["mean"],
            "reward_std": reward["std"],
        }


class ActionTelemetryAccumulator:
    """Aggregate PPO action, quote, and outcome diagnostics without step traces."""

    def __init__(self, *, max_quantile_samples: int = 8192) -> None:
        if max_quantile_samples <= 0:
            raise ValueError("max_quantile_samples must be positive")
        self.max_quantile_samples = int(max_quantile_samples)

        self._policy_sample_count = 0
        self._deterministic: bool | None = None
        self._mixed_deterministic = False
        self._enable_threshold: float | None = None
        self._mixed_enable_threshold = False
        self._enable_prob = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in ENABLE_COMPONENT_NAMES
        }
        self._enable_logit = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in ENABLE_COMPONENT_NAMES
        }
        self._enable_logit_seen = False
        self._enable_margin = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in ("bid_enabled", "ask_enabled")
        }
        self._near_threshold_counts = {
            name: {"within_0_01": 0, "within_0_05": 0}
            for name in ("bid_enabled", "ask_enabled")
        }
        self._continuous_mean = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in CONTINUOUS_COMPONENT_NAMES
        }
        self._sampled_or_chosen_continuous = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in CONTINUOUS_COMPONENT_NAMES
        }
        self._continuous_log_std = {
            name: _ScalarAccumulator(self.max_quantile_samples)
            for name in CONTINUOUS_COMPONENT_NAMES
        }
        self._entropy = _ScalarAccumulator(self.max_quantile_samples)

        self._requested_sample_count = 0
        self._requested_bid_enabled_count = 0
        self._requested_ask_enabled_count = 0
        self._requested_bid_cancel_guard_count = 0
        self._requested_ask_cancel_guard_count = 0
        self._requested_mode_counts = {name: 0 for name in QUOTE_MODE_NAMES}

        self._effective_sample_count = 0
        self._quote_bid_enabled_count = 0
        self._quote_ask_enabled_count = 0
        self._effective_mode_counts = {name: 0 for name in QUOTE_MODE_NAMES}
        self._bid_disabled_reason_counts: dict[str, int] = {}
        self._ask_disabled_reason_counts: dict[str, int] = {}
        self._quote_bid_qty = _ScalarAccumulator(self.max_quantile_samples)
        self._quote_ask_qty = _ScalarAccumulator(self.max_quantile_samples)
        self._quote_bid_offset_ticks = _ScalarAccumulator(self.max_quantile_samples)
        self._quote_ask_offset_ticks = _ScalarAccumulator(self.max_quantile_samples)

        self._outcomes = {
            name: _OutcomeByModeAccumulator(self.max_quantile_samples)
            for name in QUOTE_MODE_NAMES
        }
        self._training = {
            name: _TrainingByModeAccumulator(self.max_quantile_samples)
            for name in QUOTE_MODE_NAMES
        }

    def update_policy_action(
        self,
        policy_action: object,
        *,
        deterministic: bool,
        enable_threshold: float,
    ) -> None:
        action = _as_2d_float_array(getattr(policy_action, "action"), width=8, name="policy_action.action")
        enable_prob = _as_2d_float_array(
            getattr(policy_action, "enable_prob"),
            width=4,
            name="policy_action.enable_prob",
        )
        continuous_mean = _as_2d_float_array(
            getattr(policy_action, "continuous_mean"),
            width=4,
            name="policy_action.continuous_mean",
        )
        continuous_log_std = _as_2d_float_array(
            getattr(policy_action, "continuous_log_std"),
            width=4,
            name="policy_action.continuous_log_std",
        )
        entropy = _flatten_finite(getattr(policy_action, "entropy"), name="policy_action.entropy")
        if not (
            action.shape[0]
            == enable_prob.shape[0]
            == continuous_mean.shape[0]
            == continuous_log_std.shape[0]
            == entropy.shape[0]
        ):
            raise ValueError("policy telemetry batch dimensions must match")

        self._policy_sample_count += int(action.shape[0])
        deterministic = bool(deterministic)
        if self._deterministic is None:
            self._deterministic = deterministic
        elif self._deterministic != deterministic:
            self._mixed_deterministic = True

        threshold = float(enable_threshold)
        if not math.isfinite(threshold):
            raise ValueError("enable_threshold must be finite")
        if self._enable_threshold is None:
            self._enable_threshold = threshold
        elif not math.isclose(self._enable_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
            self._mixed_enable_threshold = True

        for idx, name in enumerate(ENABLE_COMPONENT_NAMES):
            self._enable_prob[name].update_many(enable_prob[:, idx], name=f"enable_prob.{name}")
        for idx, name in enumerate(("bid_enabled", "ask_enabled")):
            margin = enable_prob[:, idx] - threshold
            abs_margin = np.abs(margin)
            self._enable_margin[name].update_many(margin, name=f"enable_margin.{name}")
            self._near_threshold_counts[name]["within_0_01"] += int(np.count_nonzero(abs_margin <= 0.01))
            self._near_threshold_counts[name]["within_0_05"] += int(np.count_nonzero(abs_margin <= 0.05))

        enable_logits = getattr(policy_action, "enable_logits", None)
        if enable_logits is not None:
            logits = _as_2d_float_array(enable_logits, width=4, name="policy_action.enable_logits")
            if logits.shape[0] != action.shape[0]:
                raise ValueError("enable_logits batch dimension must match action")
            self._enable_logit_seen = True
            for idx, name in enumerate(ENABLE_COMPONENT_NAMES):
                self._enable_logit[name].update_many(logits[:, idx], name=f"enable_logit.{name}")

        continuous = action[:, 4:]
        for idx, name in enumerate(CONTINUOUS_COMPONENT_NAMES):
            self._continuous_mean[name].update_many(continuous_mean[:, idx], name=f"continuous_mean.{name}")
            self._sampled_or_chosen_continuous[name].update_many(continuous[:, idx], name=f"continuous.{name}")
            self._continuous_log_std[name].update_many(continuous_log_std[:, idx], name=f"continuous_log_std.{name}")
        self._entropy.update_many(entropy, name="entropy")

    def update_requested_actions(self, actions: object) -> np.ndarray:
        action_array = _as_2d_float_array(actions, width=8, name="actions")
        flags = action_array[:, :4] >= 0.5
        self._requested_sample_count += int(action_array.shape[0])
        self._requested_bid_enabled_count += int(np.count_nonzero(flags[:, 0]))
        self._requested_ask_enabled_count += int(np.count_nonzero(flags[:, 1]))
        self._requested_bid_cancel_guard_count += int(np.count_nonzero(flags[:, 2]))
        self._requested_ask_cancel_guard_count += int(np.count_nonzero(flags[:, 3]))

        mode_ids = np.zeros((action_array.shape[0],), dtype=np.int8)
        for row, (bid_enabled, ask_enabled) in enumerate(flags[:, :2]):
            mode = quote_mode_from_bools(bool(bid_enabled), bool(ask_enabled))
            self._requested_mode_counts[mode] += 1
            mode_ids[row] = _MODE_TO_ID[mode]
        return mode_ids

    def update_execution_step(self, step: ExecutionStepResult) -> int:
        if not isinstance(step, ExecutionStepResult):
            raise TypeError("step must be ExecutionStepResult")
        info = step.info or {}
        bid_enabled = _bool_value(info.get("quote_bid_enabled"), False)
        ask_enabled = _bool_value(info.get("quote_ask_enabled"), False)
        mode = quote_mode_from_bools(bid_enabled, ask_enabled)
        mode_id = _MODE_TO_ID[mode]

        self._effective_sample_count += 1
        self._quote_bid_enabled_count += int(bid_enabled)
        self._quote_ask_enabled_count += int(ask_enabled)
        self._effective_mode_counts[mode] += 1

        if bid_enabled:
            self._quote_bid_qty.update(info.get("quote_bid_qty"))
            self._quote_bid_offset_ticks.update(info.get("quote_bid_offset_ticks"))
        else:
            reason = info.get("quote_bid_disabled_reason")
            key = reason if isinstance(reason, str) and reason else "unknown"
            self._bid_disabled_reason_counts[key] = self._bid_disabled_reason_counts.get(key, 0) + 1
        if ask_enabled:
            self._quote_ask_qty.update(info.get("quote_ask_qty"))
            self._quote_ask_offset_ticks.update(info.get("quote_ask_offset_ticks"))
        else:
            reason = info.get("quote_ask_disabled_reason")
            key = reason if isinstance(reason, str) and reason else "unknown"
            self._ask_disabled_reason_counts[key] = self._ask_disabled_reason_counts.get(key, 0) + 1

        self._outcomes[mode].update(step)
        return mode_id

    def update_training_by_effective_quote_mode(
        self,
        effective_mode_ids: object,
        *,
        advantages: object,
        returns: object,
        values: object,
        rewards: object,
        valid_mask: object | None = None,
    ) -> None:
        modes = np.asarray(effective_mode_ids, dtype=np.int64).reshape(-1)
        adv = _flatten_finite(advantages, name="advantages")
        ret = _flatten_finite(returns, name="returns")
        val = _flatten_finite(values, name="values")
        rew = _flatten_finite(rewards, name="rewards")
        if not (modes.size == adv.size == ret.size == val.size == rew.size):
            raise ValueError("training telemetry arrays must have the same flattened size")
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
            if mask.size != modes.size:
                raise ValueError("valid_mask must have the same flattened size")
            modes = modes[mask]
            adv = adv[mask]
            ret = ret[mask]
            val = val[mask]
            rew = rew[mask]
        for mode_id, mode in enumerate(QUOTE_MODE_NAMES):
            mask = modes == mode_id
            if not np.any(mask):
                continue
            self._training[mode].update_many(
                advantages=adv[mask],
                returns=ret[mask],
                values=val[mask],
                rewards=rew[mask],
            )

    def _policy_distribution_dict(self) -> dict[str, object]:
        near_denominator = max(self._policy_sample_count, 1)
        payload: dict[str, object] = {
            "sample_count": int(self._policy_sample_count),
            "deterministic": None if self._mixed_deterministic else self._deterministic,
            "enable_threshold": None if self._mixed_enable_threshold else self._enable_threshold,
            "enable_prob": {
                name: stat.as_dict() for name, stat in self._enable_prob.items()
            },
            "enable_margin_to_threshold": {
                name: stat.as_dict() for name, stat in self._enable_margin.items()
            },
            "near_threshold_fraction": {
                name: {
                    key: float(count / near_denominator)
                    for key, count in counts.items()
                }
                for name, counts in self._near_threshold_counts.items()
            },
            "continuous_mean": {
                name: stat.as_dict() for name, stat in self._continuous_mean.items()
            },
            "sampled_or_chosen_continuous": {
                name: stat.as_dict()
                for name, stat in self._sampled_or_chosen_continuous.items()
            },
            "continuous_log_std": {
                name: stat.as_dict() for name, stat in self._continuous_log_std.items()
            },
            "entropy": self._entropy.as_dict(),
        }
        if self._enable_logit_seen:
            payload["enable_logit"] = {
                name: stat.as_dict() for name, stat in self._enable_logit.items()
            }
        return payload

    def _requested_actions_dict(self) -> dict[str, object]:
        denominator = max(self._requested_sample_count, 1)
        payload = {
            "requested_bid_enabled_count": int(self._requested_bid_enabled_count),
            "requested_bid_enabled_rate": float(self._requested_bid_enabled_count / denominator),
            "requested_ask_enabled_count": int(self._requested_ask_enabled_count),
            "requested_ask_enabled_rate": float(self._requested_ask_enabled_count / denominator),
            "requested_two_sided_count": int(self._requested_mode_counts[TWO_SIDED]),
            "requested_two_sided_rate": float(self._requested_mode_counts[TWO_SIDED] / denominator),
            "requested_bid_only_count": int(self._requested_mode_counts[BID_ONLY]),
            "requested_bid_only_rate": float(self._requested_mode_counts[BID_ONLY] / denominator),
            "requested_ask_only_count": int(self._requested_mode_counts[ASK_ONLY]),
            "requested_ask_only_rate": float(self._requested_mode_counts[ASK_ONLY] / denominator),
            "requested_no_quote_count": int(self._requested_mode_counts[NO_QUOTE]),
            "requested_no_quote_rate": float(self._requested_mode_counts[NO_QUOTE] / denominator),
            "requested_bid_cancel_guard_count": int(self._requested_bid_cancel_guard_count),
            "requested_bid_cancel_guard_rate": float(self._requested_bid_cancel_guard_count / denominator),
            "requested_ask_cancel_guard_count": int(self._requested_ask_cancel_guard_count),
            "requested_ask_cancel_guard_rate": float(self._requested_ask_cancel_guard_count / denominator),
        }
        return payload

    def _effective_quotes_dict(self) -> dict[str, object]:
        denominator = max(self._effective_sample_count, 1)
        return {
            "quote_bid_enabled_count": int(self._quote_bid_enabled_count),
            "quote_bid_enabled_rate": float(self._quote_bid_enabled_count / denominator),
            "quote_ask_enabled_count": int(self._quote_ask_enabled_count),
            "quote_ask_enabled_rate": float(self._quote_ask_enabled_count / denominator),
            "quote_two_sided_count": int(self._effective_mode_counts[TWO_SIDED]),
            "quote_two_sided_rate": float(self._effective_mode_counts[TWO_SIDED] / denominator),
            "quote_bid_only_count": int(self._effective_mode_counts[BID_ONLY]),
            "quote_bid_only_rate": float(self._effective_mode_counts[BID_ONLY] / denominator),
            "quote_ask_only_count": int(self._effective_mode_counts[ASK_ONLY]),
            "quote_ask_only_rate": float(self._effective_mode_counts[ASK_ONLY] / denominator),
            "quote_no_quote_count": int(self._effective_mode_counts[NO_QUOTE]),
            "quote_no_quote_rate": float(self._effective_mode_counts[NO_QUOTE] / denominator),
            "bid_disabled_reason_counts": dict(self._bid_disabled_reason_counts),
            "ask_disabled_reason_counts": dict(self._ask_disabled_reason_counts),
            "quote_bid_qty": self._quote_bid_qty.as_dict(),
            "quote_ask_qty": self._quote_ask_qty.as_dict(),
            "quote_bid_offset_ticks": self._quote_bid_offset_ticks.as_dict(),
            "quote_ask_offset_ticks": self._quote_ask_offset_ticks.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        sample_count = max(
            self._policy_sample_count,
            self._requested_sample_count,
            self._effective_sample_count,
        )
        payload: dict[str, object] = {
            "schema": ACTION_TELEMETRY_SCHEMA,
            "sample_count": int(sample_count),
            "policy_distribution": self._policy_distribution_dict(),
            "requested_actions": self._requested_actions_dict(),
            "effective_quotes": self._effective_quotes_dict(),
            "outcomes_by_effective_quote_mode": {
                mode: self._outcomes[mode].as_dict(total_steps=self._effective_sample_count)
                for mode in QUOTE_MODE_NAMES
            },
        }
        if any(item.count > 0 for item in self._training.values()):
            payload["training_by_effective_quote_mode"] = {
                mode: self._training[mode].as_dict()
                for mode in QUOTE_MODE_NAMES
            }
        return payload


def action_telemetry_brief(telemetry: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(telemetry, Mapping):
        return {}
    policy = telemetry.get("policy_distribution")
    requested = telemetry.get("requested_actions")
    effective = telemetry.get("effective_quotes")
    outcomes = telemetry.get("outcomes_by_effective_quote_mode")
    training = telemetry.get("training_by_effective_quote_mode")
    policy = policy if isinstance(policy, Mapping) else {}
    requested = requested if isinstance(requested, Mapping) else {}
    effective = effective if isinstance(effective, Mapping) else {}
    outcomes = outcomes if isinstance(outcomes, Mapping) else {}
    training = training if isinstance(training, Mapping) else {}

    enable_prob = policy.get("enable_prob")
    enable_prob = enable_prob if isinstance(enable_prob, Mapping) else {}

    def _mean_for(component: str) -> object:
        stats = enable_prob.get(component)
        if isinstance(stats, Mapping):
            return stats.get("mean")
        return None

    def _mode_value(section: Mapping[str, object], mode: str, key: str) -> object:
        item = section.get(mode)
        if isinstance(item, Mapping):
            return item.get(key)
        return None

    return {
        "bid_enable_prob_mean": _mean_for("bid_enabled"),
        "ask_enable_prob_mean": _mean_for("ask_enabled"),
        "bid_enable_requested_rate": requested.get("requested_bid_enabled_rate"),
        "ask_enable_requested_rate": requested.get("requested_ask_enabled_rate"),
        "quote_no_quote_rate": effective.get("quote_no_quote_rate"),
        "quote_bid_only_rate": effective.get("quote_bid_only_rate"),
        "quote_ask_only_rate": effective.get("quote_ask_only_rate"),
        "quote_two_sided_rate": effective.get("quote_two_sided_rate"),
        "reward_mean_by_mode": {
            mode: _mode_value(outcomes, mode, "reward_mean") for mode in QUOTE_MODE_NAMES
        },
        "advantage_mean_by_mode": {
            mode: _mode_value(training, mode, "advantage_mean") for mode in QUOTE_MODE_NAMES
        },
        "fill_step_rate_by_mode": {
            mode: _mode_value(outcomes, mode, "fill_step_rate") for mode in QUOTE_MODE_NAMES
        },
    }


__all__ = [
    "ACTION_TELEMETRY_SCHEMA",
    "NO_QUOTE",
    "BID_ONLY",
    "ASK_ONLY",
    "TWO_SIDED",
    "QUOTE_MODE_NAMES",
    "QuoteMode",
    "quote_mode_from_bools",
    "quote_mode_id_from_bools",
    "quote_mode_from_id",
    "scalar_stats",
    "ActionTelemetryAccumulator",
    "action_telemetry_brief",
]
