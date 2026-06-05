"""Training orchestration for execution PPO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mmrt.execution.env import ExecutionEnv
from mmrt.rl.normalization import ObservationNormalizer, ObservationNormalizerConfig
from mmrt.rl.ppo import PPOConfig, PPOUpdateStats, update_ppo
from mmrt.rl.rollout import RolloutBatch, RolloutCollector, RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork


__all__ = [
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
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


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

    start_event_index: int | None = None
    seed: int | None = None

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
            "start_event_index",
            _optional_nonnegative_int(self.start_event_index, "start_event_index"),
        )
        object.__setattr__(
            self,
            "seed",
            _optional_nonnegative_int(self.seed, "seed"),
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
        "start_event_index": config.start_event_index,
        "seed": config.seed,
        "use_observation_normalizer": bool(config.use_observation_normalizer),
        "network_config": {
            "hidden_sizes": list(network_config.hidden_sizes),
            "activation": network_config.activation,
            "layer_norm": bool(network_config.layer_norm),
            "orthogonal_init": bool(network_config.orthogonal_init),
            "policy_log_std_init": float(network_config.policy_log_std_init),
            "policy_log_std_min": float(network_config.policy_log_std_min),
            "policy_log_std_max": float(network_config.policy_log_std_max),
            "policy_head_gain": float(network_config.policy_head_gain),
            "value_head_gain": float(network_config.value_head_gain),
        },
        "rollout_config": {
            "rollout_steps": int(rollout_config.rollout_steps),
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

    ppo: PPOUpdateStats

    def as_dict(self) -> dict[str, object]:
        return {
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
            "ppo": self.ppo.as_dict(),
        }


class PPOTrainingResult(NamedTuple):
    policy: ActorCriticNetwork
    optimizer: torch.optim.Optimizer
    observation_normalizer: ObservationNormalizer | None
    history: tuple[PPOTrainingIterationStats, ...]
    updates_completed: int
    config: PPOTrainingConfig

    def summary_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "updates_completed": int(self.updates_completed),
            "config": training_config_to_dict(self.config),
            "history": [item.as_dict() for item in self.history],
            "final": self.history[-1].as_dict() if self.history else None,
        }


def _rollout_stats(
    update_index: int,
    batch: RolloutBatch,
    ppo_stats: PPOUpdateStats,
) -> PPOTrainingIterationStats:
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(ppo_stats, PPOUpdateStats):
        raise TypeError("ppo_stats must be a PPOUpdateStats")

    update_index = _require_nonnegative_int(update_index, "update_index")
    rewards = batch.rewards.detach()
    returns = batch.returns.detach()
    advantages = batch.advantages.detach()

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
        ppo=ppo_stats,
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


def _infer_obs_dim(env: ExecutionEnv) -> int:
    if not isinstance(env, ExecutionEnv):
        raise TypeError("env must be an ExecutionEnv")
    return _require_positive_int(int(env.config.observation_schema.dim), "obs_dim")


def train_ppo_policy(
    env: ExecutionEnv,
    *,
    config: PPOTrainingConfig = PPOTrainingConfig(),
    policy: ActorCriticNetwork | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    observation_normalizer: ObservationNormalizer | None = None,
) -> PPOTrainingResult:
    if not isinstance(env, ExecutionEnv):
        raise TypeError("env must be an ExecutionEnv")
    if not isinstance(config, PPOTrainingConfig):
        raise TypeError("config must be a PPOTrainingConfig")
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

    obs_dim = _infer_obs_dim(env)
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
        env,
        policy,
        config=config.rollout_config,
        observation_normalizer=observation_normalizer,
    )

    policy.train()
    history: list[PPOTrainingIterationStats] = []

    for update_index in range(config.num_updates):
        batch = collector.collect(
            start_event_index=config.start_event_index if update_index == 0 else None
        )
        ppo_stats = update_ppo(
            policy,
            optimizer,
            batch,
            config=config.ppo_config,
        )
        history.append(_rollout_stats(update_index, batch, ppo_stats))

    return PPOTrainingResult(
        policy=policy,
        optimizer=optimizer,
        observation_normalizer=observation_normalizer,
        history=tuple(history),
        updates_completed=len(history),
        config=config,
    )


def make_training_checkpoint_payload(result: PPOTrainingResult) -> dict[str, object]:
    if not isinstance(result, PPOTrainingResult):
        raise TypeError("result must be a PPOTrainingResult")

    return {
        "schema_version": "mmrt_execution_ppo_checkpoint_v1",
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
