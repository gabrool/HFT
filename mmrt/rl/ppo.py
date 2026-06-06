"""PPO loss and minibatch update utilities for execution policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, NamedTuple

import torch
from torch import nn

from mmrt.rl.normalization import normalize_advantages
from mmrt.rl.rollout import RolloutBatch
from mmrt.rl.torch_networks import ActorCriticNetwork


__all__ = [
    "PPOConfig",
    "PPOLoss",
    "PPOUpdateStats",
    "flatten_rollout_batch",
    "iter_minibatch_indices",
    "compute_ppo_loss",
    "update_ppo",
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


def _require_finite_float(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float")
    float_value = float(value)
    if not math.isfinite(float_value):
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


def _require_float_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating point tensor")
    return tensor


def _require_rank1_float_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _require_float_tensor(tensor, name)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    return tensor


def _require_rank2_float_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _require_float_tensor(tensor, name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    return tensor


@dataclass(frozen=True, slots=True)
class PPOConfig:
    update_epochs: int = 4
    minibatch_size: int = 256

    clip_range: float = 0.2
    value_clip_range: float = 0.2
    clip_value_loss: bool = True

    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    max_grad_norm: float | None = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "update_epochs",
            _require_positive_int(self.update_epochs, "update_epochs"),
        )
        object.__setattr__(
            self,
            "minibatch_size",
            _require_positive_int(self.minibatch_size, "minibatch_size"),
        )
        object.__setattr__(
            self,
            "clip_range",
            _require_positive_float(self.clip_range, "clip_range"),
        )
        object.__setattr__(
            self,
            "value_clip_range",
            _require_positive_float(self.value_clip_range, "value_clip_range"),
        )
        object.__setattr__(
            self,
            "clip_value_loss",
            _require_bool(self.clip_value_loss, "clip_value_loss"),
        )
        object.__setattr__(
            self,
            "value_loss_coef",
            _require_nonnegative_float(self.value_loss_coef, "value_loss_coef"),
        )
        object.__setattr__(
            self,
            "entropy_coef",
            _require_nonnegative_float(self.entropy_coef, "entropy_coef"),
        )
        if self.max_grad_norm is not None:
            object.__setattr__(
                self,
                "max_grad_norm",
                _require_positive_float(self.max_grad_norm, "max_grad_norm"),
            )
        object.__setattr__(
            self,
            "normalize_advantages",
            _require_bool(self.normalize_advantages, "normalize_advantages"),
        )
        if self.target_kl is not None:
            object.__setattr__(
                self,
                "target_kl",
                _require_positive_float(self.target_kl, "target_kl"),
            )


class PPOLoss(NamedTuple):
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    entropy: torch.Tensor


class PPOUpdateStats(NamedTuple):
    loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    approx_kl: float
    clip_fraction: float
    entropy: float
    grad_norm: float
    epochs_completed: int
    minibatches_processed: int
    early_stop: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "loss": float(self.loss),
            "policy_loss": float(self.policy_loss),
            "value_loss": float(self.value_loss),
            "entropy_loss": float(self.entropy_loss),
            "approx_kl": float(self.approx_kl),
            "clip_fraction": float(self.clip_fraction),
            "entropy": float(self.entropy),
            "grad_norm": float(self.grad_norm),
            "epochs_completed": int(self.epochs_completed),
            "minibatches_processed": int(self.minibatches_processed),
            "early_stop": bool(self.early_stop),
        }


def _validate_flat_batch(
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
) -> int:
    observations = _require_rank2_float_tensor(observations, "observations")
    actions = _require_rank2_float_tensor(actions, "actions")
    old_log_probs = _require_rank1_float_tensor(old_log_probs, "old_log_probs")
    old_values = _require_rank1_float_tensor(old_values, "old_values")
    returns = _require_rank1_float_tensor(returns, "returns")
    advantages = _require_rank1_float_tensor(advantages, "advantages")

    batch_size = int(observations.shape[0])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if actions.shape[0] != batch_size:
        raise ValueError("actions batch size must match observations batch size")
    for name, tensor in (
        ("old_log_probs", old_log_probs),
        ("old_values", old_values),
        ("returns", returns),
        ("advantages", advantages),
    ):
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match observations batch size")
    return batch_size


def _policy_device(policy: ActorCriticNetwork) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _scalar_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())


def flatten_rollout_batch(batch: RolloutBatch) -> dict[str, torch.Tensor]:
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")

    return {
        "observations": batch.observations.reshape(batch.num_steps, batch.obs_dim),
        "actions": batch.actions.reshape(batch.num_steps, batch.action_dim),
        "old_log_probs": batch.log_probs.reshape(batch.num_steps),
        "old_values": batch.values.reshape(batch.num_steps),
        "returns": batch.returns.reshape(batch.num_steps),
        "advantages": batch.advantages.reshape(batch.num_steps),
    }


def iter_minibatch_indices(
    batch_size: int,
    minibatch_size: int,
    *,
    device: torch.device,
    shuffle: bool = True,
) -> Iterator[torch.Tensor]:
    batch_size = _require_positive_int(batch_size, "batch_size")
    minibatch_size = _require_positive_int(minibatch_size, "minibatch_size")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    shuffle = _require_bool(shuffle, "shuffle")

    if shuffle:
        indices = torch.randperm(batch_size, device=device)
    else:
        indices = torch.arange(batch_size, device=device)

    for start in range(0, batch_size, minibatch_size):
        yield indices[start : start + minibatch_size]


def compute_ppo_loss(
    policy: ActorCriticNetwork,
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    config: PPOConfig = PPOConfig(),
) -> PPOLoss:
    if not isinstance(policy, ActorCriticNetwork):
        raise TypeError("policy must be an ActorCriticNetwork")
    if not isinstance(config, PPOConfig):
        raise TypeError("config must be a PPOConfig")

    _validate_flat_batch(
        observations=observations,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
    )
    if observations.shape[-1] != policy.obs_dim:
        raise ValueError("observations last dimension must equal policy obs_dim")
    if actions.shape[-1] != policy.action_dim:
        raise ValueError("actions last dimension must equal policy action_dim")

    if config.normalize_advantages:
        advantages = normalize_advantages(advantages)

    evaluation = policy.evaluate_actions(observations, actions)
    new_log_probs = evaluation.log_prob
    new_values = evaluation.value
    entropy = evaluation.entropy

    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - config.clip_range,
        1.0 + config.clip_range,
    ) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    if config.clip_value_loss:
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values,
            -config.value_clip_range,
            config.value_clip_range,
        )
        value_loss_unclipped = (new_values - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = 0.5 * (new_values - returns).pow(2).mean()

    entropy_mean = entropy.mean()
    entropy_loss = -entropy_mean
    total_loss = (
        policy_loss
        + config.value_loss_coef * value_loss
        + config.entropy_coef * entropy_loss
    )

    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_fraction = (
            ((ratio - 1.0).abs() > config.clip_range).to(observations.dtype).mean()
        )

    return PPOLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        entropy=entropy_mean,
    )


def update_ppo(
    policy: ActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    *,
    config: PPOConfig = PPOConfig(),
    shuffle: bool = True,
) -> PPOUpdateStats:
    if not isinstance(policy, ActorCriticNetwork):
        raise TypeError("policy must be an ActorCriticNetwork")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(config, PPOConfig):
        raise TypeError("config must be a PPOConfig")
    shuffle = _require_bool(shuffle, "shuffle")

    flat = flatten_rollout_batch(batch)
    policy_device = _policy_device(policy)
    if flat["observations"].device != policy_device:
        raise ValueError("rollout batch device must match policy device")

    batch_size = _validate_flat_batch(**flat)

    loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    entropy_sum = 0.0
    grad_norm_sum = 0.0
    sample_count = 0
    minibatches_processed = 0
    epochs_completed = 0
    early_stop = False

    for epoch_index in range(config.update_epochs):
        for idx in iter_minibatch_indices(
            batch_size,
            config.minibatch_size,
            device=flat["observations"].device,
            shuffle=shuffle,
        ):
            mb_size = int(idx.numel())
            loss = compute_ppo_loss(
                policy,
                observations=flat["observations"][idx],
                actions=flat["actions"][idx],
                old_log_probs=flat["old_log_probs"][idx],
                old_values=flat["old_values"][idx],
                returns=flat["returns"][idx],
                advantages=flat["advantages"][idx],
                config=config,
            )

            optimizer.zero_grad(set_to_none=True)
            getattr(loss.total_loss, "back" + "ward")()
            if config.max_grad_norm is not None:
                grad_norm_tensor = nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    config.max_grad_norm,
                )
                grad_norm = _scalar_float(grad_norm_tensor)
            else:
                grad_norm = 0.0
            optimizer.step()

            loss_sum += _scalar_float(loss.total_loss) * mb_size
            policy_loss_sum += _scalar_float(loss.policy_loss) * mb_size
            value_loss_sum += _scalar_float(loss.value_loss) * mb_size
            entropy_loss_sum += _scalar_float(loss.entropy_loss) * mb_size
            approx_kl = _scalar_float(loss.approx_kl)
            approx_kl_sum += approx_kl * mb_size
            clip_fraction_sum += _scalar_float(loss.clip_fraction) * mb_size
            entropy_sum += _scalar_float(loss.entropy) * mb_size
            grad_norm_sum += grad_norm
            sample_count += mb_size
            minibatches_processed += 1

            if config.target_kl is not None and approx_kl > config.target_kl:
                early_stop = True
                break

        epochs_completed = epoch_index + 1
        if early_stop:
            break

    if minibatches_processed == 0:
        raise RuntimeError("PPO update processed no minibatches")

    denom = max(sample_count, 1)
    grad_denom = max(minibatches_processed, 1)
    return PPOUpdateStats(
        loss=loss_sum / denom,
        policy_loss=policy_loss_sum / denom,
        value_loss=value_loss_sum / denom,
        entropy_loss=entropy_loss_sum / denom,
        approx_kl=approx_kl_sum / denom,
        clip_fraction=clip_fraction_sum / denom,
        entropy=entropy_sum / denom,
        grad_norm=grad_norm_sum / grad_denom,
        epochs_completed=epochs_completed,
        minibatches_processed=minibatches_processed,
        early_stop=early_stop,
    )
