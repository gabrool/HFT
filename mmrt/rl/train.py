"""Training orchestration for execution PPO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import NamedTuple, Sequence

import torch

from mmrt.execution.env import ExecutionEnv
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.rl.action_telemetry import ActionTelemetryAccumulator, action_telemetry_brief
from mmrt.rl.device import resolve_torch_device
from mmrt.rl.normalization import ObservationNormalizer, ObservationNormalizerConfig
from mmrt.rl.ppo import PPOConfig, PPOUpdateStats, update_ppo
from mmrt.rl.rollout import TRAIN_WINDOW_SAMPLING_MODES, RolloutBatch, RolloutCollector, RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork


PPO_CHECKPOINT_SCHEMA = "mmrt_execution_ppo_checkpoint"


__all__ = [
    "PPO_CHECKPOINT_SCHEMA",
    "PPOTrainingConfig",
    "PPOTrainingIterationStats",
    "PPOTrainingResult",
    "create_policy",
    "create_optimizer",
    "train_ppo_policy",
    "training_config_to_dict",
    "make_training_checkpoint_payload",
]


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


def _require_finite_float(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float")
    float_value = float(value)
    if float_value != float_value or float_value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite float")
    return float_value


def _require_positive_float(value: float, name: str) -> float:
    float_value = _require_finite_float(value, name)
    if float_value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return float_value


def _require_nonnegative_float(value: float, name: str) -> float:
    float_value = _require_finite_float(value, name)
    if float_value < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite float")
    return float_value


def _resolve_device(device: str | torch.device | None) -> torch.device:
    return resolve_torch_device(device)


def _scalar_float(tensor: torch.Tensor) -> float:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    return float(tensor.detach().cpu().item())


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    num_updates: int = 10

    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0

    seed: int | None = None
    train_window_sampling: str = "stratified_random"

    use_observation_normalizer: bool = True

    network_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    observation_normalizer_config: ObservationNormalizerConfig = field(
        default_factory=ObservationNormalizerConfig
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_updates",
            _require_positive_int(self.num_updates, "num_updates"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _require_positive_float(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(
            self,
            "adam_eps",
            _require_positive_float(self.adam_eps, "adam_eps"),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _require_nonnegative_float(self.weight_decay, "weight_decay"),
        )
        object.__setattr__(
            self,
            "seed",
            _optional_nonnegative_int(self.seed, "seed"),
        )
        object.__setattr__(
            self,
            "train_window_sampling",
            _coerce_train_window_sampling(self.train_window_sampling),
        )
        object.__setattr__(
            self,
            "use_observation_normalizer",
            _require_bool(self.use_observation_normalizer, "use_observation_normalizer"),
        )
        if not isinstance(self.network_config, ActorCriticConfig):
            raise TypeError("network_config must be an ActorCriticConfig")
        if not isinstance(self.rollout_config, RolloutConfig):
            raise TypeError("rollout_config must be a RolloutConfig")
        if not isinstance(self.ppo_config, PPOConfig):
            raise TypeError("ppo_config must be a PPOConfig")
        effective_batch_size = self.rollout_config.rollout_steps * self.rollout_config.num_envs
        if self.ppo_config.minibatch_size > effective_batch_size:
            raise ValueError("minibatch_size must be <= rollout_steps * num_envs")
        if not isinstance(
            self.observation_normalizer_config,
            ObservationNormalizerConfig,
        ):
            raise TypeError(
                "observation_normalizer_config must be an ObservationNormalizerConfig"
            )


def training_config_to_dict(config: PPOTrainingConfig) -> dict[str, object]:
    if not isinstance(config, PPOTrainingConfig):
        raise TypeError("config must be a PPOTrainingConfig")

    network_config = config.network_config
    rollout_config = config.rollout_config
    ppo_config = config.ppo_config
    observation_normalizer_config = config.observation_normalizer_config

    return {
        "num_updates": int(config.num_updates),
        "learning_rate": float(config.learning_rate),
        "adam_eps": float(config.adam_eps),
        "weight_decay": float(config.weight_decay),
        "seed": config.seed,
        "train_window_sampling": config.train_window_sampling,
        "use_observation_normalizer": bool(config.use_observation_normalizer),
        "network_config": {
            "hidden_sizes": list(network_config.hidden_sizes),
            "activation": network_config.activation,
            "layer_norm": bool(network_config.layer_norm),
            "orthogonal_init": bool(network_config.orthogonal_init),
            "enable_threshold": float(network_config.enable_threshold),
            "enable_logit_bias_init": float(network_config.enable_logit_bias_init),
            "continuous_log_std_init": float(network_config.continuous_log_std_init),
            "continuous_log_std_min": float(network_config.continuous_log_std_min),
            "continuous_log_std_max": float(network_config.continuous_log_std_max),
            "policy_head_gain": float(network_config.policy_head_gain),
            "value_head_gain": float(network_config.value_head_gain),
        },
        "rollout_config": {
            "rollout_steps": int(rollout_config.rollout_steps),
            "num_envs": int(rollout_config.num_envs),
            "effective_batch_size": int(rollout_config.rollout_steps * rollout_config.num_envs),
            "gamma": float(rollout_config.gamma),
            "gae_lambda": float(rollout_config.gae_lambda),
            "deterministic": bool(rollout_config.deterministic),
            "reset_on_terminal": bool(rollout_config.reset_on_terminal),
            "device": None if rollout_config.device is None else str(rollout_config.device),
            "dtype": str(rollout_config.dtype),
        },
        "ppo_config": {
            "update_epochs": int(ppo_config.update_epochs),
            "minibatch_size": int(ppo_config.minibatch_size),
            "clip_range": float(ppo_config.clip_range),
            "value_clip_range": float(ppo_config.value_clip_range),
            "clip_value_loss": bool(ppo_config.clip_value_loss),
            "value_loss_coef": float(ppo_config.value_loss_coef),
            "entropy_coef": float(ppo_config.entropy_coef),
            "max_grad_norm": (
                None if ppo_config.max_grad_norm is None else float(ppo_config.max_grad_norm)
            ),
            "normalize_advantages": bool(ppo_config.normalize_advantages),
            "target_kl": None if ppo_config.target_kl is None else float(ppo_config.target_kl),
        },
        "observation_normalizer_config": {
            "enabled": bool(observation_normalizer_config.enabled),
            "update": bool(observation_normalizer_config.update),
            "epsilon": float(observation_normalizer_config.epsilon),
            "clip": (
                None
                if observation_normalizer_config.clip is None
                else float(observation_normalizer_config.clip)
            ),
            "rms_epsilon": float(observation_normalizer_config.rms_epsilon),
        },
    }


class PPOTrainingIterationStats(NamedTuple):
    update_index: int

    rollout_reward_sum: float
    rollout_reward_mean: float
    rollout_reward_std: float
    rollout_return_mean: float
    rollout_advantage_mean: float
    rollout_done_count: int
    rollout_truncated_count: int
    rollout_episode_count: int
    sampling_stats: dict[str, object]

    ppo: PPOUpdateStats
    rollout_seconds: float
    env_step_seconds: float
    policy_forward_seconds: float
    ppo_update_seconds: float
    update_total_seconds: float
    env_steps_per_sec: float
    policy_forward_steps_per_sec: float
    total_steps_per_sec: float
    telemetry: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "update_index": int(self.update_index),
            "rollout": {
                "reward_sum": float(self.rollout_reward_sum),
                "reward_mean": float(self.rollout_reward_mean),
                "reward_std": float(self.rollout_reward_std),
                "return_mean": float(self.rollout_return_mean),
                "advantage_mean": float(self.rollout_advantage_mean),
                "done_count": int(self.rollout_done_count),
                "truncated_count": int(self.rollout_truncated_count),
                "episode_count": int(self.rollout_episode_count),
            },
            "sampling": dict(self.sampling_stats),
            "ppo": self.ppo.as_dict(),
            "timing": {
                "rollout_seconds": float(self.rollout_seconds),
                "env_step_seconds": float(self.env_step_seconds),
                "policy_forward_seconds": float(self.policy_forward_seconds),
                "ppo_update_seconds": float(self.ppo_update_seconds),
                "update_total_seconds": float(self.update_total_seconds),
                "env_steps_per_sec": float(self.env_steps_per_sec),
                "policy_forward_steps_per_sec": float(self.policy_forward_steps_per_sec),
                "total_steps_per_sec": float(self.total_steps_per_sec),
            },
        }
        if self.telemetry is not None:
            payload["telemetry_brief"] = action_telemetry_brief(self.telemetry)
        return payload


class PPOTrainingResult(NamedTuple):
    policy: ActorCriticNetwork
    optimizer: torch.optim.Optimizer
    observation_normalizer: ObservationNormalizer | None
    history: tuple[PPOTrainingIterationStats, ...]
    updates_completed: int
    config: PPOTrainingConfig
    sampling_stats: dict[str, object] | None = None
    telemetry_aggregate: dict[str, object] | None = None

    def summary_dict(self) -> dict[str, object]:
        history = [item.as_dict() for item in self.history]
        final = None
        if self.history:
            final = self.history[-1].as_dict()
            if self.history[-1].telemetry is not None:
                final["telemetry"] = dict(self.history[-1].telemetry)
        return {
            "status": "ok",
            "updates_completed": int(self.updates_completed),
            "config": training_config_to_dict(self.config),
            "sampling": None if self.sampling_stats is None else dict(self.sampling_stats),
            "history": history,
            "final": final,
            "telemetry_aggregate": (
                None if self.telemetry_aggregate is None else dict(self.telemetry_aggregate)
            ),
        }


def _rollout_stats(
    update_index: int,
    batch: RolloutBatch,
    ppo_stats: PPOUpdateStats,
    *,
    ppo_update_seconds: float,
) -> PPOTrainingIterationStats:
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(ppo_stats, PPOUpdateStats):
        raise TypeError("ppo_stats must be a PPOUpdateStats")

    update_index = _require_nonnegative_int(update_index, "update_index")
    rewards = batch.rewards.detach()
    returns = batch.returns.detach()
    advantages = batch.advantages.detach()
    timing = batch.timing or {}
    rollout_seconds = float(timing.get("rollout_seconds", 0.0))
    ppo_update_seconds = float(ppo_update_seconds)
    update_total_seconds = rollout_seconds + ppo_update_seconds
    total_steps = float(batch.num_steps)

    return PPOTrainingIterationStats(
        update_index=update_index,
        rollout_reward_sum=_scalar_float(rewards.sum()),
        rollout_reward_mean=_scalar_float(rewards.mean()),
        rollout_reward_std=_scalar_float(rewards.std(unbiased=False)),
        rollout_return_mean=_scalar_float(returns.mean()),
        rollout_advantage_mean=_scalar_float(advantages.mean()),
        rollout_done_count=int(batch.dones.detach().to(torch.int64).sum().cpu().item()),
        rollout_truncated_count=int(
            batch.truncated.detach().to(torch.int64).sum().cpu().item()
        ),
        rollout_episode_count=int(batch.episode_count),
        sampling_stats=dict(batch.sampling_stats or {}),
        ppo=ppo_stats,
        rollout_seconds=rollout_seconds,
        env_step_seconds=float(timing.get("env_step_seconds", 0.0)),
        policy_forward_seconds=float(timing.get("policy_forward_seconds", 0.0)),
        ppo_update_seconds=ppo_update_seconds,
        update_total_seconds=update_total_seconds,
        env_steps_per_sec=float(timing.get("env_steps_per_sec", 0.0)),
        policy_forward_steps_per_sec=float(timing.get("policy_forward_steps_per_sec", 0.0)),
        total_steps_per_sec=float(total_steps / update_total_seconds) if update_total_seconds > 0.0 else 0.0,
        telemetry=batch.telemetry,
    )


def create_policy(
    *,
    obs_dim: int,
    config: PPOTrainingConfig,
) -> ActorCriticNetwork:
    obs_dim = _require_positive_int(obs_dim, "obs_dim")
    if not isinstance(config, PPOTrainingConfig):
        raise TypeError("config must be a PPOTrainingConfig")

    policy = ActorCriticNetwork(obs_dim=obs_dim, config=config.network_config)
    device = _resolve_device(config.rollout_config.device)
    return policy.to(device=device, dtype=config.rollout_config.dtype)


def create_optimizer(
    policy: ActorCriticNetwork,
    *,
    config: PPOTrainingConfig,
) -> torch.optim.Optimizer:
    if not isinstance(policy, ActorCriticNetwork):
        raise TypeError("policy must be an ActorCriticNetwork")
    if not isinstance(config, PPOTrainingConfig):
        raise TypeError("config must be a PPOTrainingConfig")

    return torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )


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


def _infer_obs_dim(envs: Sequence[ExecutionEnv]) -> int:
    if not envs:
        raise ValueError("envs must be non-empty")
    obs_dim = _require_positive_int(int(envs[0].config.observation_schema.dim), "obs_dim")
    for item in envs[1:]:
        if int(item.config.observation_schema.dim) != obs_dim:
            raise ValueError("all env observation dimensions must match")
    return obs_dim


def train_ppo_policy(
    env: ExecutionEnv | Sequence[ExecutionEnv],
    *,
    config: PPOTrainingConfig = PPOTrainingConfig(),
    policy: ActorCriticNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    observation_normalizer: ObservationNormalizer | None = None,
    decision_row_ranges: Sequence[DecisionSplitRange] | None = None,
    start_decision_rows: Sequence[int] | None = None,
) -> PPOTrainingResult:
    envs = _coerce_envs(env)
    if not isinstance(config, PPOTrainingConfig):
        raise TypeError("config must be a PPOTrainingConfig")
    if len(envs) != config.rollout_config.num_envs:
        raise ValueError("config.rollout_config.num_envs must match the number of envs")
    if policy is not None and not isinstance(policy, ActorCriticNetwork):
        raise TypeError("policy must be None or an ActorCriticNetwork")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be None or a torch.optim.Optimizer")
    if observation_normalizer is not None and not isinstance(
        observation_normalizer,
        ObservationNormalizer,
    ):
        raise TypeError("observation_normalizer must be None or ObservationNormalizer")

    if config.seed is not None:
        torch.manual_seed(config.seed)

    obs_dim = _infer_obs_dim(envs)
    device = _resolve_device(config.rollout_config.device)

    if policy is None:
        policy = create_policy(obs_dim=obs_dim, config=config)
    else:
        if policy.obs_dim != obs_dim:
            raise ValueError("policy obs_dim must match env observation dim")
        policy.to(device=device, dtype=config.rollout_config.dtype)

    if optimizer is None:
        optimizer = create_optimizer(policy, config=config)

    if config.use_observation_normalizer:
        if observation_normalizer is None:
            observation_normalizer = ObservationNormalizer(
                obs_shape=obs_dim,
                config=config.observation_normalizer_config,
            )
        observation_normalizer.to(device=device, dtype=config.rollout_config.dtype)
    else:
        observation_normalizer = None

    collector = RolloutCollector(
        envs,
        policy,
        config=config.rollout_config,
        observation_normalizer=observation_normalizer,
        decision_row_ranges=decision_row_ranges,
        train_window_sampling=config.train_window_sampling,
        seed=config.seed,
    )

    policy.train()
    history: list[PPOTrainingIterationStats] = []
    telemetry_aggregate = ActionTelemetryAccumulator()

    for update_index in range(config.num_updates):
        batch = collector.collect(
            start_decision_rows=start_decision_rows if update_index == 0 else None,
            aggregate_telemetry=telemetry_aggregate,
        )
        ppo_started = time.perf_counter()
        ppo_stats = update_ppo(
            policy,
            optimizer,
            batch,
            config=config.ppo_config,
        )
        ppo_update_seconds = time.perf_counter() - ppo_started
        history.append(_rollout_stats(update_index, batch, ppo_stats, ppo_update_seconds=ppo_update_seconds))

    return PPOTrainingResult(
        policy=policy,
        optimizer=optimizer,
        observation_normalizer=observation_normalizer,
        history=tuple(history),
        updates_completed=len(history),
        config=config,
        sampling_stats=collector.sampling_stats(),
        telemetry_aggregate=telemetry_aggregate.as_dict(),
    )


def make_training_checkpoint_payload(result: PPOTrainingResult) -> dict[str, object]:
    if not isinstance(result, PPOTrainingResult):
        raise TypeError("result must be a PPOTrainingResult")

    return {
        "schema": PPO_CHECKPOINT_SCHEMA,
        "updates_completed": result.updates_completed,
        "config": training_config_to_dict(result.config),
        "summary": result.summary_dict(),
        "policy_state_dict": result.policy.state_dict(),
        "optimizer_state_dict": result.optimizer.state_dict(),
        "observation_normalizer_state_dict": (
            None
            if result.observation_normalizer is None
            else result.observation_normalizer.state_dict()
        ),
    }
