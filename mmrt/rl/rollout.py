"""Vectorized rollout collection and GAE computation for execution PPO."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import random
import time
from typing import Any, NamedTuple, Sequence

import torch

from mmrt.execution.env import ExecutionEnv
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.rl.action_telemetry import ActionTelemetryAccumulator
from mmrt.rl.device import canonicalize_torch_device, resolve_torch_device
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.reward_modes import (
    FifoLotLedger,
    HorizonRewardProjector,
    RewardAnchor,
    TrainingRewardConfig,
    TrainingRewardMode,
)
from mmrt.rl.torch_networks import ActorCriticNetwork


__all__ = [
    "TRAIN_WINDOW_SAMPLING_MODES",
    "TrainWindowSampler",
    "RolloutConfig",
    "RolloutBatch",
    "compute_discount_factors_from_dt_us",
    "compute_gae",
    "RolloutCollector",
    "collect_rollout",
]


TRAIN_WINDOW_SAMPLING_MODES = ("stratified_random", "cyclic_spread", "chronological")
DISCOUNT_MODES = ("step", "time")


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive int")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative int")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


def _coerce_train_window_sampling(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("train_window_sampling must be str")
    normalized = value.strip()
    if normalized not in TRAIN_WINDOW_SAMPLING_MODES:
        raise ValueError(f"train_window_sampling must be one of {TRAIN_WINDOW_SAMPLING_MODES}")
    return normalized


def _coerce_discount_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("discount_mode must be str")
    normalized = value.strip()
    if normalized not in DISCOUNT_MODES:
        raise ValueError('discount_mode must be "step" or "time"')
    return normalized


def _require_probability(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a probability in [0, 1]")
    float_value = float(value)
    if not 0.0 <= float_value <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return float_value


def _stats_from_tensor(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must be non-empty")
    values = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _dt_stats_from_tensor(tensor: torch.Tensor) -> dict[str, float]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.numel() <= 0:
        raise ValueError("tensor must be non-empty")
    values = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    quantiles = torch.quantile(
        values,
        torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64),
    )
    return {
        "step_dt_us_min": float(values.min().item()),
        "step_dt_us_mean": float(values.mean().item()),
        "step_dt_us_p50": float(quantiles[0].item()),
        "step_dt_us_p90": float(quantiles[1].item()),
        "step_dt_us_p99": float(quantiles[2].item()),
        "step_dt_us_max": float(values.max().item()),
    }


def _require_float_dtype(dtype: torch.dtype, name: str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if not torch.empty((), dtype=dtype).dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating point dtype")
    return dtype


def _policy_device(policy: ActorCriticNetwork) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_device(
    policy: ActorCriticNetwork,
    device: str | torch.device | None,
) -> torch.device:
    if device is not None:
        return resolve_torch_device(device)
    return _policy_device(policy)


def _observation_to_tensor(
    obs: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    obs_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    obs_dim = _require_positive_int(obs_dim, "obs_dim")
    dtype = _require_float_dtype(dtype, "dtype")
    device = canonicalize_torch_device(device)
    tensor = torch.as_tensor(obs, device=device, dtype=dtype)
    if tuple(tensor.shape) != (obs_dim,):
        raise ValueError(f"observation shape must be ({obs_dim},)")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if (
            canonicalize_torch_device(out.device) != device
            or out.dtype != dtype
            or tuple(out.shape) != (obs_dim,)
        ):
            raise ValueError("out must match observation device, dtype, and shape")
        out.copy_(tensor)
        return out
    return tensor.clone()


def _observations_to_tensor(
    observations: Sequence[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    obs_dim: int,
    out: torch.Tensor,
) -> torch.Tensor:
    if len(observations) != int(out.shape[0]):
        raise ValueError("observation count must match output batch")
    for row, obs in enumerate(observations):
        _observation_to_tensor(obs, device=device, dtype=dtype, obs_dim=obs_dim, out=out[row])
    return out


class TrainWindowSampler:
    """Seedable start-row sampler over rollout-capable train split ranges."""

    def __init__(
        self,
        decision_row_ranges: Sequence[DecisionSplitRange],
        *,
        mode: str = "stratified_random",
        seed: int | None = None,
        num_envs: int = 1,
    ) -> None:
        ranges = tuple(decision_row_ranges)
        if not ranges:
            raise ValueError("decision_row_ranges must be non-empty")
        if any(not isinstance(item, DecisionSplitRange) for item in ranges):
            raise TypeError("decision_row_ranges entries must be DecisionSplitRange")
        if any(item.rollout_step_capacity <= 0 for item in ranges):
            raise ValueError("decision_row_ranges must each contain at least two decision rows")
        self.ranges = ranges
        self.mode = _coerce_train_window_sampling(mode)
        self.seed = _optional_nonnegative_int(seed, "seed")
        self.num_envs = _require_positive_int(num_envs, "num_envs")
        self._rng = random.Random(self.seed)
        self._reset_counts = [0 for _ in range(self.num_envs)]
        self._chronological_cursor = 0
        cumulative: list[int] = []
        running = 0
        for entry in ranges:
            running += int(entry.rollout_step_capacity)
            cumulative.append(running)
        self._cumulative_capacity = tuple(cumulative)
        self.total_capacity_rows = int(running)
        self.train_row_count = int(sum(entry.row_count for entry in ranges))
        self._sampled_start_rows: list[int] = []
        self._sampled_range_indices: list[int] = []

    @property
    def sampled_start_count(self) -> int:
        return len(self._sampled_start_rows)

    def range_for_row(self, row: int) -> DecisionSplitRange | None:
        row = _require_nonnegative_int(row, "row")
        for entry in self.ranges:
            if entry.start_decision_row <= row and row + 1 < entry.end_decision_row:
                return entry
        return None

    def _range_index_for(self, split_range: DecisionSplitRange) -> int:
        for index, entry in enumerate(self.ranges):
            if (
                entry.role == split_range.role
                and entry.segment_key == split_range.segment_key
                and entry.start_decision_row == split_range.start_decision_row
                and entry.end_decision_row == split_range.end_decision_row
            ):
                return index
        raise ValueError("split_range is not managed by this sampler")

    def _sample_at_capacity_position(self, position: int) -> tuple[int, DecisionSplitRange, int]:
        if self.total_capacity_rows <= 0:
            raise RuntimeError("train window sampler has no rollout-capable rows")
        normalized = int(position) % self.total_capacity_rows
        range_index = bisect_right(self._cumulative_capacity, normalized)
        previous_capacity = 0 if range_index == 0 else self._cumulative_capacity[range_index - 1]
        selected = self.ranges[range_index]
        offset = normalized - previous_capacity
        return selected.start_decision_row + offset, selected, range_index

    def _sample_position(self, env_index: int) -> int:
        env_index = _require_nonnegative_int(env_index, "env_index")
        if env_index >= self.num_envs:
            raise ValueError("env_index must be < num_envs")
        reset_round = self._reset_counts[env_index]
        total = self.total_capacity_rows
        if self.mode == "stratified_random":
            stratum_count = max(1, self.num_envs)
            stratum = (env_index + reset_round) % stratum_count
            lo = int(stratum * total / stratum_count)
            hi = int((stratum + 1) * total / stratum_count)
            lo = min(lo, total - 1)
            hi = max(lo + 1, min(hi, total))
            return self._rng.randrange(lo, hi)
        if self.mode == "cyclic_spread":
            position = int((env_index + 0.5) * total / max(1, self.num_envs))
            return (position + reset_round) % total
        if self.mode == "chronological":
            position = self._chronological_cursor
            self._chronological_cursor += 1
            return position
        raise RuntimeError("unreachable train window sampling mode")

    def sample(self, env_index: int) -> tuple[int, DecisionSplitRange]:
        position = self._sample_position(env_index)
        row, selected, range_index = self._sample_at_capacity_position(position)
        self._reset_counts[env_index] += 1
        self.record(row, selected, range_index=range_index)
        return row, selected

    def record(
        self,
        row: int,
        split_range: DecisionSplitRange,
        *,
        range_index: int | None = None,
    ) -> None:
        row = _require_nonnegative_int(row, "row")
        if not (split_range.start_decision_row <= row and row + 1 < split_range.end_decision_row):
            raise ValueError("sampled start row must lie inside a rollout-capable split range")
        if range_index is None:
            range_index = self._range_index_for(split_range)
        self._sampled_start_rows.append(row)
        self._sampled_range_indices.append(int(range_index))

    def stats_since(self, start_index: int = 0) -> dict[str, object]:
        start_index = _require_nonnegative_int(start_index, "start_index")
        rows = self._sampled_start_rows[start_index:]
        range_indices = self._sampled_range_indices[start_index:]
        stats: dict[str, object] = {
            "train_window_sampling": self.mode,
            "seed": self.seed,
            "sampled_start_count": len(rows),
            "unique_train_ranges_visited": len(set(range_indices)),
            "train_row_count": self.train_row_count,
            "train_capacity_rows": self.total_capacity_rows,
            "estimated_train_coverage_fraction": (
                float(len(set(rows)) / self.train_row_count) if self.train_row_count > 0 else 0.0
            ),
        }
        if rows:
            stats.update(
                {
                    "sampled_start_decision_row_min": int(min(rows)),
                    "sampled_start_decision_row_max": int(max(rows)),
                    "sampled_start_decision_row_mean": float(sum(rows) / len(rows)),
                }
            )
        else:
            stats.update(
                {
                    "sampled_start_decision_row_min": None,
                    "sampled_start_decision_row_max": None,
                    "sampled_start_decision_row_mean": None,
                }
            )
        return stats


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    rollout_steps: int = 1024
    num_envs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    discount_mode: str = "step"
    discount_horizon_us: int = 1_000_000
    deterministic: bool = False
    reset_on_terminal: bool = True
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32
    reward_config: TrainingRewardConfig = TrainingRewardConfig()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_steps",
            _require_positive_int(self.rollout_steps, "rollout_steps"),
        )
        object.__setattr__(self, "num_envs", _require_positive_int(self.num_envs, "num_envs"))
        object.__setattr__(self, "gamma", _require_probability(self.gamma, "gamma"))
        object.__setattr__(
            self,
            "gae_lambda",
            _require_probability(self.gae_lambda, "gae_lambda"),
        )
        object.__setattr__(self, "discount_mode", _coerce_discount_mode(self.discount_mode))
        object.__setattr__(
            self,
            "discount_horizon_us",
            _require_positive_int(self.discount_horizon_us, "discount_horizon_us"),
        )
        object.__setattr__(
            self,
            "deterministic",
            _require_bool(self.deterministic, "deterministic"),
        )
        object.__setattr__(
            self,
            "reset_on_terminal",
            _require_bool(self.reset_on_terminal, "reset_on_terminal"),
        )
        if self.device is not None and not isinstance(self.device, (str, torch.device)):
            raise TypeError("device must be None, str, or torch.device")
        object.__setattr__(self, "dtype", _require_float_dtype(self.dtype, "dtype"))
        if not isinstance(self.reward_config, TrainingRewardConfig):
            raise TypeError("reward_config must be a TrainingRewardConfig")


class RolloutBatch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    entropies: torch.Tensor
    episode_count: int
    decision_rows: torch.Tensor | None = None
    step_dt_us: torch.Tensor | None = None
    discounts: torch.Tensor | None = None
    lambda_discounts: torch.Tensor | None = None
    timing: dict[str, object] | None = None
    sampling_stats: dict[str, object] | None = None
    telemetry: dict[str, object] | None = None
    env_rewards: torch.Tensor | None = None
    projected_rewards: torch.Tensor | None = None
    reward_valid_mask: torch.Tensor | None = None
    reward_components: dict[str, torch.Tensor] | None = None
    reward_mode: str = TrainingRewardMode.EQUITY_DELTA.value
    reward_projection_stats: dict[str, object] | None = None

    @property
    def num_steps(self) -> int:
        if self.rewards.ndim == 1:
            return int(self.rewards.shape[0])
        return int(self.rewards.shape[0] * self.rewards.shape[1])

    @property
    def rollout_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def num_envs(self) -> int:
        return 1 if self.rewards.ndim == 1 else int(self.rewards.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def to(self, device: str | torch.device) -> "RolloutBatch":
        decision_rows = self.decision_rows
        step_dt_us = self.step_dt_us
        discounts = self.discounts
        lambda_discounts = self.lambda_discounts
        reward_components = self.reward_components
        return RolloutBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            terminated=self.terminated.to(device),
            truncated=self.truncated.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            entropies=self.entropies.to(device),
            episode_count=self.episode_count,
            decision_rows=None if decision_rows is None else decision_rows.to(device),
            step_dt_us=None if step_dt_us is None else step_dt_us.to(device),
            discounts=None if discounts is None else discounts.to(device),
            lambda_discounts=None if lambda_discounts is None else lambda_discounts.to(device),
            timing=None if self.timing is None else dict(self.timing),
            sampling_stats=None if self.sampling_stats is None else dict(self.sampling_stats),
            telemetry=None if self.telemetry is None else dict(self.telemetry),
            env_rewards=None if self.env_rewards is None else self.env_rewards.to(device),
            projected_rewards=None if self.projected_rewards is None else self.projected_rewards.to(device),
            reward_valid_mask=None if self.reward_valid_mask is None else self.reward_valid_mask.to(device),
            reward_components=(
                None
                if reward_components is None
                else {name: tensor.to(device) for name, tensor in reward_components.items()}
            ),
            reward_mode=self.reward_mode,
            reward_projection_stats=(
                None
                if self.reward_projection_stats is None
                else dict(self.reward_projection_stats)
            ),
        )


def _require_reward_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim not in (1, 2):
        raise ValueError(f"{name} must be rank-1 or rank-2")
    return tensor


def compute_discount_factors_from_dt_us(
    dt_us: torch.Tensor,
    *,
    factor_at_horizon: float,
    horizon_us: int,
) -> torch.Tensor:
    if not isinstance(dt_us, torch.Tensor):
        raise TypeError("dt_us must be a torch.Tensor")
    horizon_us = _require_positive_int(horizon_us, "horizon_us")
    factor_at_horizon = _require_probability(factor_at_horizon, "factor_at_horizon")
    dt = dt_us.to(dtype=torch.float32 if not dt_us.dtype.is_floating_point else dt_us.dtype)
    if not torch.isfinite(dt).all():
        raise ValueError("dt_us must contain finite values")
    if bool((dt < 0).any().cpu().item()):
        raise ValueError("dt_us must be nonnegative")
    return torch.pow(
        torch.as_tensor(factor_at_horizon, dtype=dt.dtype, device=dt.device),
        dt / float(horizon_us),
    )


def _coerce_discount_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    rewards: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype is torch.bool:
        raise TypeError(f"{name} must be floating point or numeric")
    if tensor.shape != rewards.shape:
        raise ValueError(f"{name} must have the same shape as rewards")
    out = tensor.to(device=rewards.device, dtype=rewards.dtype)
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} must contain finite values")
    if bool(((out < 0.0) | (out > 1.0)).any().cpu().item()):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return out


def _coerce_valid_mask(
    tensor: torch.Tensor | None,
    *,
    rewards: torch.Tensor,
) -> torch.Tensor:
    if tensor is None:
        return torch.ones_like(rewards, dtype=torch.bool)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if tensor.shape != rewards.shape:
        raise ValueError("valid_mask must have the same shape as rewards")
    if tensor.dtype is not torch.bool:
        raise TypeError("valid_mask must be a bool tensor")
    return tensor.to(device=rewards.device)


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor | float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    discounts: torch.Tensor | None = None,
    lambda_discounts: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = _require_reward_tensor(rewards, "rewards")
    values = _require_reward_tensor(values, "values")
    dones = _require_reward_tensor(dones, "dones")
    if not rewards.dtype.is_floating_point:
        raise TypeError("rewards must be a floating point tensor")
    if not values.dtype.is_floating_point:
        raise TypeError("values must be a floating point tensor")
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")
    if rewards.shape != dones.shape:
        raise ValueError("dones must have the same shape as rewards")
    if dones.dtype is not torch.bool:
        raise TypeError("dones must be a bool tensor")
    gamma = _require_probability(gamma, "gamma")
    gae_lambda = _require_probability(gae_lambda, "gae_lambda")
    discount_tensor = (
        None
        if discounts is None
        else _coerce_discount_tensor(discounts, name="discounts", rewards=rewards)
    )
    lambda_discount_tensor = (
        None
        if lambda_discounts is None
        else _coerce_discount_tensor(lambda_discounts, name="lambda_discounts", rewards=rewards)
    )
    valid = _coerce_valid_mask(valid_mask, rewards=rewards)

    advantages = torch.empty_like(rewards)
    last_value_tensor = torch.as_tensor(last_value, dtype=rewards.dtype, device=rewards.device)
    if rewards.ndim == 1:
        if last_value_tensor.numel() != 1:
            raise ValueError("last_value must be scalar for rank-1 rewards")
        next_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        last_value_tensor = last_value_tensor.reshape(())
    else:
        if tuple(last_value_tensor.shape) != (rewards.shape[1],):
            raise ValueError("last_value must have shape (num_envs,) for rank-2 rewards")
        next_gae = torch.zeros((rewards.shape[1],), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        next_value = last_value_tensor if t == rewards.shape[0] - 1 else values[t + 1]
        if t == rewards.shape[0] - 1:
            next_valid = torch.ones_like(valid[t], dtype=rewards.dtype)
        else:
            next_valid = valid[t + 1].to(dtype=rewards.dtype)
        valid_t = valid[t]
        nonterminal = (~dones[t]).to(dtype=rewards.dtype) * next_valid
        discount_t = gamma if discount_tensor is None else discount_tensor[t]
        lambda_discount_t = gae_lambda if lambda_discount_tensor is None else lambda_discount_tensor[t]
        delta = rewards[t] + discount_t * next_value * nonterminal - values[t]
        next_gae = delta + discount_t * lambda_discount_t * nonterminal * next_gae
        next_gae = torch.where(valid_t, next_gae, torch.zeros_like(next_gae))
        advantages[t] = next_gae

    returns = advantages + values
    return advantages, returns


def _coerce_envs(env: ExecutionEnv | Sequence[ExecutionEnv]) -> tuple[ExecutionEnv, ...]:
    if isinstance(env, ExecutionEnv):
        return (env,)
    if isinstance(env, (str, bytes)):
        raise TypeError("env must be ExecutionEnv or a sequence of ExecutionEnv")
    envs = tuple(env)
    if not envs:
        raise ValueError("envs must be non-empty")
    for item in envs:
        if not isinstance(item, ExecutionEnv):
            raise TypeError("all envs must be ExecutionEnv")
    return envs


class RolloutCollector:
    def __init__(
        self,
        env: ExecutionEnv | Sequence[ExecutionEnv],
        policy: ActorCriticNetwork,
        *,
        config: RolloutConfig = RolloutConfig(),
        observation_normalizer: ObservationNormalizer | None = None,
        decision_row_ranges: Sequence[DecisionSplitRange] | None = None,
        train_window_sampling: str = "stratified_random",
        seed: int | None = None,
    ) -> None:
        envs = _coerce_envs(env)
        if not isinstance(policy, ActorCriticNetwork):
            raise TypeError("policy must be an ActorCriticNetwork")
        if not isinstance(config, RolloutConfig):
            raise TypeError("config must be a RolloutConfig")
        if len(envs) != config.num_envs:
            raise ValueError("config.num_envs must match the number of envs")
        if observation_normalizer is not None and not isinstance(
            observation_normalizer,
            ObservationNormalizer,
        ):
            raise TypeError("observation_normalizer must be None or ObservationNormalizer")

        self.envs = envs
        self.policy = policy
        self.config = config
        self.observation_normalizer = observation_normalizer
        self.device = _resolve_device(policy, config.device)
        self.dtype = config.dtype
        self.num_envs = len(envs)

        for item in envs:
            if int(item.config.observation_schema.dim) != policy.obs_dim:
                raise ValueError("all env observation dimensions must match policy.obs_dim")

        if decision_row_ranges is None:
            ranges: tuple[DecisionSplitRange, ...] = ()
            sampler = None
        else:
            ranges = tuple(decision_row_ranges)
            if not ranges:
                raise ValueError("decision_row_ranges must be non-empty when provided")
            if any(not isinstance(item, DecisionSplitRange) for item in ranges):
                raise TypeError("decision_row_ranges entries must be DecisionSplitRange")
            if any(item.rollout_step_capacity <= 0 for item in ranges):
                raise ValueError("decision_row_ranges must each contain at least two decision rows")
            sampler = TrainWindowSampler(
                ranges,
                mode=train_window_sampling,
                seed=seed,
                num_envs=self.num_envs,
            )
        self.decision_row_ranges = ranges
        self._train_window_sampler = sampler

        self._current_observations: list[Any | None] = [None for _ in envs]
        self._current_decision_rows = [-1 for _ in envs]
        self._current_ranges: list[DecisionSplitRange | None] = [None for _ in envs]
        self._obs_scratch = torch.empty((self.num_envs, self.policy.obs_dim), device=self.device, dtype=self.dtype)
        self._has_reset = False
        self.episode_count = 0
        self._reward_episode_ids = [-1 for _ in envs]
        self._lot_ledgers: list[FifoLotLedger | None] = [None for _ in envs]

    def _mid_price_for_decision_row(self, env: ExecutionEnv, row: int) -> float:
        row = _require_nonnegative_int(row, "row")
        if row >= env.decision_grid.n_rows:
            raise ValueError("row must be < decision_grid.n_rows")
        book_ptr = int(env.decision_grid.book_ptr[row])
        l2 = env.tape.arrays.l2_events[book_ptr]
        tick_size = float(env.tape.manifest.symbol_spec.tick_size)
        return float((int(l2["best_bid_tick"]) + int(l2["best_ask_tick"])) * tick_size * 0.5)

    def _ledger_for_env(self, env_index: int) -> FifoLotLedger:
        ledger = self._lot_ledgers[env_index]
        if ledger is None:
            env = self.envs[env_index]
            symbol_spec = env.tape.manifest.symbol_spec
            ledger = FifoLotLedger(
                tick_size=float(symbol_spec.tick_size),
                contract_size=float(symbol_spec.contract_size),
                qty_epsilon=float(env.config.fill_simulator_config.qty_epsilon),
            )
            self._lot_ledgers[env_index] = ledger
        return ledger

    def reset(self, *, start_decision_rows: Sequence[int] | None = None) -> torch.Tensor:
        if start_decision_rows is not None and len(start_decision_rows) != self.num_envs:
            raise ValueError("start_decision_rows length must match num_envs")
        for env_index in range(self.num_envs):
            requested = None if start_decision_rows is None else int(start_decision_rows[env_index])
            self._reset_one(env_index, start_decision_row=requested)
        self._has_reset = True
        return self._current_obs_tensor()

    def _range_for_row(self, row: int) -> DecisionSplitRange | None:
        row = _require_nonnegative_int(row, "row")
        if self._train_window_sampler is not None:
            return self._train_window_sampler.range_for_row(row)
        if not self.decision_row_ranges:
            return None
        for entry in self.decision_row_ranges:
            if entry.start_decision_row <= row and row + 1 < entry.end_decision_row:
                return entry
        return None

    def _next_start_for_env(self, env_index: int) -> tuple[int | None, DecisionSplitRange | None]:
        if self._train_window_sampler is None:
            return None, None
        return self._train_window_sampler.sample(env_index)

    def _reset_one(self, env_index: int, *, start_decision_row: int | None = None) -> None:
        env = self.envs[env_index]
        if start_decision_row is None:
            start_decision_row, selected_range = self._next_start_for_env(env_index)
        else:
            selected_range = self._range_for_row(start_decision_row)
            if self.decision_row_ranges and selected_range is None:
                raise ValueError("start_decision_row must lie inside a rollout split range")

        reset = env.reset(start_decision_row=start_decision_row)
        decision_row = int(reset.info["decision_grid_row_index"])
        current_range = selected_range if selected_range is not None else self._range_for_row(decision_row)
        if self.decision_row_ranges and current_range is None:
            raise ValueError("environment reset outside rollout split ranges")
        if start_decision_row is not None and self._train_window_sampler is not None:
            self._train_window_sampler.record(decision_row, current_range)  # type: ignore[arg-type]
        self._current_observations[env_index] = reset.observation
        self._current_decision_rows[env_index] = decision_row
        self._current_ranges[env_index] = current_range
        self._reward_episode_ids[env_index] += 1
        ledger = self._ledger_for_env(env_index)
        ledger.reset(
            initial_inventory_qty=float(env.config.initial_position.inventory_qty),
            entry_price=self._mid_price_for_decision_row(env, decision_row),
        )

    def _can_step(self, env_index: int) -> bool:
        current_range = self._current_ranges[env_index]
        if current_range is None:
            return True
        return self._current_decision_rows[env_index] + 1 < current_range.end_decision_row

    def _current_obs_tensor(self) -> torch.Tensor:
        if not self._has_reset and any(obs is None for obs in self._current_observations):
            raise RuntimeError("environment has not been reset")
        observations = [obs for obs in self._current_observations]
        if any(obs is None for obs in observations):
            raise RuntimeError("environment has not been reset")
        return _observations_to_tensor(
            observations,  # type: ignore[arg-type]
            device=self.device,
            dtype=self.dtype,
            obs_dim=self.policy.obs_dim,
            out=self._obs_scratch,
        )

    def _normalize_obs_for_policy(self, obs: torch.Tensor, *, update: bool) -> torch.Tensor:
        if self.observation_normalizer is None:
            return obs
        if update:
            return self.observation_normalizer.update_and_normalize(obs)
        return self.observation_normalizer.normalize(obs)

    def collect(
        self,
        *,
        start_decision_rows: Sequence[int] | None = None,
        aggregate_telemetry: ActionTelemetryAccumulator | None = None,
    ) -> RolloutBatch:
        if aggregate_telemetry is not None and not isinstance(
            aggregate_telemetry,
            ActionTelemetryAccumulator,
        ):
            raise TypeError("aggregate_telemetry must be ActionTelemetryAccumulator or None")
        config = self.config
        sample_stat_start = (
            self._train_window_sampler.sampled_start_count
            if self._train_window_sampler is not None
            else 0
        )
        if not self._has_reset or start_decision_rows is not None:
            self.reset(start_decision_rows=start_decision_rows)

        T = config.rollout_steps
        N = self.num_envs
        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim
        device = self.device
        dtype = self.dtype

        observations = torch.empty((T, N, obs_dim), device=device, dtype=dtype)
        actions = torch.empty((T, N, action_dim), device=device, dtype=dtype)
        log_probs = torch.empty((T, N), device=device, dtype=dtype)
        values = torch.empty((T, N), device=device, dtype=dtype)
        rewards = torch.empty((T, N), device=device, dtype=dtype)
        entropies = torch.empty((T, N), device=device, dtype=dtype)
        decision_rows = torch.empty((T, N), device=device, dtype=torch.int64)
        step_dt_us = torch.empty((T, N), device=device, dtype=torch.int64)
        requested_mode_ids = torch.empty((T, N), dtype=torch.int64)
        effective_mode_ids = torch.empty((T, N), dtype=torch.int64)

        dones = torch.empty((T, N), device=device, dtype=torch.bool)
        terminated = torch.empty((T, N), device=device, dtype=torch.bool)
        truncated = torch.empty((T, N), device=device, dtype=torch.bool)

        episode_count_start = self.episode_count
        policy_forward_seconds = 0.0
        env_step_seconds = 0.0
        rollout_started = time.perf_counter()
        rollout_telemetry = ActionTelemetryAccumulator()
        telemetry_targets = (
            (rollout_telemetry, aggregate_telemetry)
            if aggregate_telemetry is not None
            else (rollout_telemetry,)
        )
        reward_config = config.reward_config
        projector = HorizonRewardProjector.from_execution(
            decision_grid=self.envs[0].decision_grid,
            tape=self.envs[0].tape,
            config=reward_config,
        )
        anchors: list[RewardAnchor] = []
        equity_by_episode: dict[tuple[int, int], dict[int, float]] = {}
        tail_step_counts = [0 for _ in range(N)]

        for t in range(T):
            for env_index in range(N):
                if not self._can_step(env_index):
                    self._reset_one(env_index, start_decision_row=None)

            raw_obs = self._current_obs_tensor()
            obs_for_policy = self._normalize_obs_for_policy(raw_obs, update=True)
            observations[t].copy_(obs_for_policy)
            decision_rows[t].copy_(torch.as_tensor(self._current_decision_rows, device=device, dtype=torch.int64))

            policy_started = time.perf_counter()
            with torch.inference_mode():
                policy_out = self.policy.sample_action(
                    obs_for_policy,
                    deterministic=config.deterministic,
                )
            policy_forward_seconds += time.perf_counter() - policy_started

            actions[t].copy_(policy_out.action)
            log_probs[t].copy_(policy_out.log_prob)
            values[t].copy_(policy_out.value)
            entropies[t].copy_(policy_out.entropy)
            for telemetry in telemetry_targets:
                telemetry.update_policy_action(
                    policy_out,
                    deterministic=config.deterministic,
                    enable_threshold=self.policy.config.enable_threshold,
                )
            requested_modes = rollout_telemetry.update_requested_actions(policy_out.action)
            if aggregate_telemetry is not None:
                aggregate_telemetry.update_requested_actions(policy_out.action)
            requested_mode_ids[t].copy_(torch.as_tensor(requested_modes, dtype=torch.int64))

            action_cpu = policy_out.action.detach().to("cpu").numpy()
            for env_index, env_action in enumerate(action_cpu):
                env = self.envs[env_index]
                current_row = int(self._current_decision_rows[env_index])
                current_ts = int(env.decision_grid.decision_local_ts_us[current_row])
                step_started = time.perf_counter()
                step = env.step(env_action)
                env_step_seconds += time.perf_counter() - step_started
                mode_id = rollout_telemetry.update_execution_step(step.execution)
                if aggregate_telemetry is not None:
                    aggregate_telemetry.update_execution_step(step.execution)
                effective_mode_ids[t, env_index] = mode_id

                rewards[t, env_index] = float(step.reward)
                terminated[t, env_index] = bool(step.done)
                split_boundary = False
                current_range = self._current_ranges[env_index]
                next_row = int(step.info["next_decision_grid_row_index"])
                next_ts = int(env.decision_grid.decision_local_ts_us[next_row])
                dt_us = next_ts - current_ts
                if dt_us < 0:
                    raise RuntimeError("decision grid local timestamps must be nondecreasing")
                step_dt_us[t, env_index] = dt_us
                if current_range is not None:
                    if next_row >= current_range.end_decision_row:
                        raise RuntimeError("rollout step crossed split boundary")
                    split_boundary = next_row + 1 >= current_range.end_decision_row
                truncated[t, env_index] = bool(step.truncated or split_boundary)
                dones[t, env_index] = bool(step.done or step.truncated or split_boundary)
                episode_id = int(self._reward_episode_ids[env_index])
                episode_equity = equity_by_episode.setdefault((env_index, episode_id), {})
                previous_equity = float(step.info["previous_equity"])
                current_equity = float(step.info["current_equity"])
                episode_equity.setdefault(current_row, previous_equity)
                episode_equity[next_row] = current_equity
                ledger = self._ledger_for_env(env_index)
                realized_lot_pnl = ledger.apply_fills(step.fills)
                range_end_row = (
                    int(env.decision_grid.n_rows)
                    if current_range is None
                    else int(current_range.end_decision_row)
                )
                anchors.append(
                    RewardAnchor(
                        t_index=t,
                        env_index=env_index,
                        episode_id=episode_id,
                        decision_row=current_row,
                        next_decision_row=next_row,
                        decision_local_ts_us=current_ts,
                        next_decision_local_ts_us=next_ts,
                        range_end_row=range_end_row,
                        previous_equity=previous_equity,
                        current_equity=current_equity,
                        env_reward=float(step.reward),
                        fills=step.fills,
                        inventory_after_step=float(step.position.inventory_qty),
                        current_mid_after_step=self._mid_price_for_decision_row(env, next_row),
                        realized_lot_pnl=realized_lot_pnl,
                    )
                )

                if bool(dones[t, env_index]):
                    self.episode_count += 1
                    if not config.reset_on_terminal:
                        raise RuntimeError("terminal reached with reset_on_terminal=False")
                    self._reset_one(env_index, start_decision_row=None)
                else:
                    self._current_observations[env_index] = step.observation
                    self._current_decision_rows[env_index] = next_row

        last_value = torch.zeros((N,), device=device, dtype=dtype)
        active_indices = [index for index in range(N) if not bool(dones[-1, index])]
        if active_indices:
            raw_obs = self._current_obs_tensor()
            obs_for_policy = self._normalize_obs_for_policy(raw_obs, update=False)
            policy_started = time.perf_counter()
            with torch.inference_mode():
                last_values = self.policy.forward(obs_for_policy).value
            policy_forward_seconds += time.perf_counter() - policy_started
            last_value.copy_(last_values)
            last_value[dones[-1]] = 0.0

        def missing_tail_envs() -> list[int]:
            missing: set[int] = set()
            for anchor in anchors:
                if anchor.episode_id != self._reward_episode_ids[anchor.env_index]:
                    continue
                episode_equity = equity_by_episode.get((anchor.env_index, anchor.episode_id), {})
                for required_row in projector.required_rows(anchor):
                    if required_row not in episode_equity:
                        missing.add(anchor.env_index)
                        break
            return sorted(missing)

        if not reward_config.is_equity_delta:
            while True:
                missing_envs = missing_tail_envs()
                if not missing_envs:
                    break
                step_envs = [env_index for env_index in missing_envs if self._can_step(env_index)]
                if not step_envs:
                    break
                raw_obs = self._current_obs_tensor()
                obs_for_policy = self._normalize_obs_for_policy(raw_obs, update=False)
                policy_started = time.perf_counter()
                with torch.inference_mode():
                    policy_out = self.policy.sample_action(
                        obs_for_policy,
                        deterministic=config.deterministic,
                    )
                policy_forward_seconds += time.perf_counter() - policy_started
                action_cpu = policy_out.action.detach().to("cpu").numpy()
                for env_index in step_envs:
                    env = self.envs[env_index]
                    current_row = int(self._current_decision_rows[env_index])
                    current_ts = int(env.decision_grid.decision_local_ts_us[current_row])
                    current_range = self._current_ranges[env_index]
                    step_started = time.perf_counter()
                    step = env.step(action_cpu[env_index])
                    env_step_seconds += time.perf_counter() - step_started
                    tail_step_counts[env_index] += 1
                    next_row = int(step.info["next_decision_grid_row_index"])
                    next_ts = int(env.decision_grid.decision_local_ts_us[next_row])
                    if next_ts < current_ts:
                        raise RuntimeError("decision grid local timestamps must be nondecreasing")
                    if current_range is not None and next_row >= current_range.end_decision_row:
                        raise RuntimeError("tail step crossed split boundary")
                    episode_id = int(self._reward_episode_ids[env_index])
                    episode_equity = equity_by_episode.setdefault((env_index, episode_id), {})
                    episode_equity.setdefault(current_row, float(step.info["previous_equity"]))
                    episode_equity[next_row] = float(step.info["current_equity"])
                    self._ledger_for_env(env_index).apply_fills(step.fills)
                    split_boundary = (
                        False
                        if current_range is None
                        else next_row + 1 >= current_range.end_decision_row
                    )
                    done = bool(step.done or step.truncated or split_boundary)
                    if done:
                        self.episode_count += 1
                        if not config.reset_on_terminal:
                            raise RuntimeError("terminal reached with reset_on_terminal=False")
                        self._reset_one(env_index, start_decision_row=None)
                    else:
                        self._current_observations[env_index] = step.observation
                        self._current_decision_rows[env_index] = next_row

        if config.discount_mode == "time":
            discounts = compute_discount_factors_from_dt_us(
                step_dt_us,
                factor_at_horizon=config.gamma,
                horizon_us=config.discount_horizon_us,
            ).to(device=device, dtype=dtype)
            lambda_discounts = compute_discount_factors_from_dt_us(
                step_dt_us,
                factor_at_horizon=config.gae_lambda,
                horizon_us=config.discount_horizon_us,
            ).to(device=device, dtype=dtype)
            discounts_for_gae: torch.Tensor | None = discounts
            lambda_discounts_for_gae: torch.Tensor | None = lambda_discounts
        else:
            discounts = torch.full_like(rewards, float(config.gamma))
            lambda_discounts = torch.full_like(rewards, float(config.gae_lambda))
            discounts_for_gae = None
            lambda_discounts_for_gae = None

        env_rewards = rewards.clone()
        projection = projector.project(anchors, equity_by_episode, shape=(T, N))
        projected_rewards = torch.as_tensor(
            projection.projected_rewards,
            device=device,
            dtype=dtype,
        )
        reward_valid_mask = torch.as_tensor(
            projection.valid_mask,
            device=device,
            dtype=torch.bool,
        )
        if not bool(reward_valid_mask.any().cpu().item()):
            raise RuntimeError(
                f"training reward mode {reward_config.mode.value!r} produced zero valid anchors"
            )
        reward_components = {
            name: torch.as_tensor(values, device=device, dtype=dtype)
            for name, values in projection.components.items()
        }
        reward_projection_stats = dict(projection.stats)
        tail_step_count = int(sum(tail_step_counts))
        reward_projection_stats.update(
            {
                "tail_step_count": tail_step_count,
                "tail_step_mean_per_env": float(tail_step_count / max(N, 1)),
                "max_tail_step_count": int(max(tail_step_counts) if tail_step_counts else 0),
            }
        )
        rewards = projected_rewards

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            discounts=discounts_for_gae,
            lambda_discounts=lambda_discounts_for_gae,
            valid_mask=reward_valid_mask,
        )
        for telemetry in telemetry_targets:
            telemetry.update_training_by_effective_quote_mode(
                effective_mode_ids,
                advantages=advantages,
                returns=returns,
                values=values,
                rewards=rewards,
                valid_mask=reward_valid_mask,
            )

        rollout_seconds = time.perf_counter() - rollout_started
        total_steps = float(T * N)
        timing: dict[str, object] = {
            "rollout_seconds": float(rollout_seconds),
            "policy_forward_seconds": float(policy_forward_seconds),
            "env_step_seconds": float(env_step_seconds),
            "env_steps_per_sec": float(total_steps / env_step_seconds) if env_step_seconds > 0.0 else 0.0,
            "policy_forward_steps_per_sec": float(total_steps / policy_forward_seconds) if policy_forward_seconds > 0.0 else 0.0,
            "total_steps_per_sec": float(total_steps / rollout_seconds) if rollout_seconds > 0.0 else 0.0,
            "discount_mode": config.discount_mode,
            "discount_horizon_us": int(config.discount_horizon_us),
        }
        timing["training_reward_mode"] = reward_config.mode.value
        timing["reward_horizon_us"] = int(reward_config.reward_horizon_us)
        timing["reward_valid_fraction"] = float(reward_projection_stats["valid_fraction"])
        timing["tail_step_count"] = int(reward_projection_stats["tail_step_count"])
        timing.update(_dt_stats_from_tensor(step_dt_us))
        timing.update(_stats_from_tensor("discount", discounts))
        timing.update(_stats_from_tensor("lambda_discount", lambda_discounts))
        sampling_stats = (
            self._train_window_sampler.stats_since(sample_stat_start)
            if self._train_window_sampler is not None
            else None
        )

        return RolloutBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            terminated=terminated,
            truncated=truncated,
            advantages=advantages,
            returns=returns,
            entropies=entropies,
            episode_count=self.episode_count - episode_count_start,
            decision_rows=decision_rows,
            step_dt_us=step_dt_us,
            discounts=discounts,
            lambda_discounts=lambda_discounts,
            timing=timing,
            sampling_stats=sampling_stats,
            telemetry=rollout_telemetry.as_dict(),
            env_rewards=env_rewards,
            projected_rewards=projected_rewards,
            reward_valid_mask=reward_valid_mask,
            reward_components=reward_components,
            reward_mode=reward_config.mode.value,
            reward_projection_stats=reward_projection_stats,
        )

    def sampling_stats(self) -> dict[str, object] | None:
        if self._train_window_sampler is None:
            return None
        return self._train_window_sampler.stats_since(0)


def collect_rollout(
    env: ExecutionEnv | Sequence[ExecutionEnv],
    policy: ActorCriticNetwork,
    *,
    config: RolloutConfig = RolloutConfig(),
    observation_normalizer: ObservationNormalizer | None = None,
    start_decision_rows: Sequence[int] | None = None,
    decision_row_ranges: Sequence[DecisionSplitRange] | None = None,
    train_window_sampling: str = "stratified_random",
    seed: int | None = None,
) -> RolloutBatch:
    collector = RolloutCollector(
        env,
        policy,
        config=config,
        observation_normalizer=observation_normalizer,
        decision_row_ranges=decision_row_ranges,
        train_window_sampling=train_window_sampling,
        seed=seed,
    )
    return collector.collect(start_decision_rows=start_decision_rows)
