"""Rollout collection and GAE computation for execution PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from mmrt.execution.env import ExecutionEnv
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.torch_networks import ActorCriticNetwork


__all__ = [
    "RolloutConfig",
    "RolloutBatch",
    "compute_gae",
    "RolloutCollector",
    "collect_rollout",
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


def _require_probability(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a probability in [0, 1]")
    float_value = float(value)
    if not 0.0 <= float_value <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return float_value


def _require_float_dtype(dtype: torch.dtype, name: str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{name} must be a torch.dtype")
    if not torch.empty((), dtype=dtype).dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating point dtype")
    return dtype


def _resolve_device(
    policy: ActorCriticNetwork,
    device: str | torch.device | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)

    try:
        parameter = next(policy.parameters())
    except StopIteration:
        return torch.device("cpu")
    return parameter.device


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
    tensor = torch.as_tensor(obs, device=device, dtype=dtype)
    if tuple(tensor.shape) != (obs_dim,):
        raise ValueError(f"observation shape must be ({obs_dim},)")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.device != device or out.dtype != dtype or tuple(out.shape) != (obs_dim,):
            raise ValueError("out must match observation device, dtype, and shape")
        out.copy_(tensor)
        return out
    return tensor.clone()


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    deterministic: bool = False
    reset_on_terminal: bool = True
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_steps",
            _require_positive_int(self.rollout_steps, "rollout_steps"),
        )
        object.__setattr__(self, "gamma", _require_probability(self.gamma, "gamma"))
        object.__setattr__(
            self,
            "gae_lambda",
            _require_probability(self.gae_lambda, "gae_lambda"),
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

    @property
    def num_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def to(self, device: str | torch.device) -> "RolloutBatch":
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
        )


def _require_rank_one_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    return tensor


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor | float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = _require_rank_one_tensor(rewards, "rewards")
    values = _require_rank_one_tensor(values, "values")
    dones = _require_rank_one_tensor(dones, "dones")
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

    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    last_value_tensor = torch.as_tensor(
        last_value,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    if last_value_tensor.numel() != 1:
        raise ValueError("last_value must be scalar")
    last_value_tensor = last_value_tensor.reshape(())

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_value = last_value_tensor
        else:
            next_value = values[t + 1]

        nonterminal = (~dones[t]).to(dtype=rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class RolloutCollector:
    def __init__(
        self,
        env: ExecutionEnv,
        policy: ActorCriticNetwork,
        *,
        config: RolloutConfig = RolloutConfig(),
        observation_normalizer: ObservationNormalizer | None = None,
    ) -> None:
        if not isinstance(env, ExecutionEnv):
            raise TypeError("env must be an ExecutionEnv")
        if not isinstance(policy, ActorCriticNetwork):
            raise TypeError("policy must be an ActorCriticNetwork")
        if not isinstance(config, RolloutConfig):
            raise TypeError("config must be a RolloutConfig")
        if observation_normalizer is not None and not isinstance(
            observation_normalizer,
            ObservationNormalizer,
        ):
            raise TypeError("observation_normalizer must be None or ObservationNormalizer")

        self.env = env
        self.policy = policy
        self.config = config
        self.observation_normalizer = observation_normalizer
        self.device = _resolve_device(policy, config.device)
        self.dtype = config.dtype

        self._current_observation: Any | None = None
        self._obs_scratch = torch.empty(self.policy.obs_dim, device=self.device, dtype=self.dtype)
        self._has_reset = False
        self.episode_count = 0

    def reset(self, *, start_event_index: int | None = None) -> torch.Tensor:
        reset = self.env.reset(start_event_index=start_event_index)
        self._current_observation = reset.observation
        self._has_reset = True
        return self._current_obs_tensor()

    def _current_obs_tensor(self) -> torch.Tensor:
        if not self._has_reset or self._current_observation is None:
            raise RuntimeError("environment has not been reset")
        return _observation_to_tensor(
            self._current_observation,
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

    def collect(self, *, start_event_index: int | None = None) -> RolloutBatch:
        config = self.config
        if not self._has_reset or start_event_index is not None:
            self.reset(start_event_index=start_event_index)

        T = config.rollout_steps
        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim
        device = self.device
        dtype = self.dtype

        observations = torch.empty((T, obs_dim), device=device, dtype=dtype)
        actions = torch.empty((T, action_dim), device=device, dtype=dtype)
        log_probs = torch.empty(T, device=device, dtype=dtype)
        values = torch.empty(T, device=device, dtype=dtype)
        rewards = torch.empty(T, device=device, dtype=dtype)
        entropies = torch.empty(T, device=device, dtype=dtype)

        dones = torch.empty(T, device=device, dtype=torch.bool)
        terminated = torch.empty(T, device=device, dtype=torch.bool)
        truncated = torch.empty(T, device=device, dtype=torch.bool)

        episode_count_start = self.episode_count

        for t in range(T):
            raw_obs = self._current_obs_tensor()
            obs_for_policy = self._normalize_obs_for_policy(raw_obs, update=True)
            observations[t].copy_(obs_for_policy)

            with torch.no_grad():
                policy_out = self.policy.sample_action(
                    obs_for_policy.unsqueeze(0),
                    deterministic=config.deterministic,
                )

            action = policy_out.action.squeeze(0)
            actions[t].copy_(action)
            log_probs[t].copy_(policy_out.log_prob.squeeze(0))
            values[t].copy_(policy_out.value.squeeze(0))
            entropies[t].copy_(policy_out.entropy.squeeze(0))

            env_action = action.detach().to("cpu").tolist()
            step = self.env.step(env_action)

            rewards[t] = float(step.reward)
            terminated[t] = bool(step.done)
            truncated[t] = bool(step.truncated)
            dones[t] = bool(step.done or step.truncated)

            if bool(dones[t]):
                self.episode_count += 1
                if not config.reset_on_terminal:
                    raise RuntimeError("terminal reached with reset_on_terminal=False")
                reset = self.env.reset()
                self._current_observation = reset.observation
            else:
                self._current_observation = step.observation

        if bool(dones[-1]):
            last_value = torch.zeros((), device=device, dtype=dtype)
        else:
            raw_obs = self._current_obs_tensor()
            obs_for_policy = self._normalize_obs_for_policy(raw_obs, update=False)
            with torch.no_grad():
                last_value = self.policy.forward(obs_for_policy.unsqueeze(0)).value.squeeze(0)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
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
        )


def collect_rollout(
    env: ExecutionEnv,
    policy: ActorCriticNetwork,
    *,
    config: RolloutConfig = RolloutConfig(),
    observation_normalizer: ObservationNormalizer | None = None,
    start_event_index: int | None = None,
) -> RolloutBatch:
    collector = RolloutCollector(
        env,
        policy,
        config=config,
        observation_normalizer=observation_normalizer,
    )
    return collector.collect(start_event_index=start_event_index)
