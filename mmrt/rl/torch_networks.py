from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn


EXECUTION_ACTION_DIM = 6

ACTION_COMPONENT_NAMES = (
    "bid_enable_logit",
    "ask_enable_logit",
    "bid_distance_raw",
    "ask_distance_raw",
    "bid_size_raw",
    "ask_size_raw",
)

LOG_2PI = math.log(2.0 * math.pi)

__all__ = [
    "EXECUTION_ACTION_DIM",
    "ACTION_COMPONENT_NAMES",
    "ActorCriticConfig",
    "ActorCriticOutput",
    "PolicyAction",
    "PolicyEvaluation",
    "build_mlp",
    "diagonal_gaussian_log_prob",
    "diagonal_gaussian_entropy",
    "ActorCriticNetwork",
]


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


def _require_float_tensor_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating point tensor")
    return tensor


def _normalize_hidden_sizes(hidden_sizes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not isinstance(hidden_sizes, (tuple, list)):
        raise TypeError("hidden_sizes must be a tuple or list of positive ints")
    return tuple(_require_positive_int(size, "hidden_sizes item") for size in hidden_sizes)


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "tanh"
    layer_norm: bool = False
    orthogonal_init: bool = True

    policy_log_std_init: float = -0.5
    policy_log_std_min: float = -5.0
    policy_log_std_max: float = 2.0

    policy_head_gain: float = 0.01
    value_head_gain: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", _normalize_hidden_sizes(self.hidden_sizes))

        if self.activation not in ("tanh", "relu", "silu"):
            raise ValueError('activation must be one of "tanh", "relu", or "silu"')
        if not isinstance(self.layer_norm, bool):
            raise TypeError("layer_norm must be bool")
        if not isinstance(self.orthogonal_init, bool):
            raise TypeError("orthogonal_init must be bool")

        object.__setattr__(
            self,
            "policy_log_std_init",
            _require_finite_float(self.policy_log_std_init, "policy_log_std_init"),
        )
        policy_log_std_min = _require_finite_float(
            self.policy_log_std_min,
            "policy_log_std_min",
        )
        policy_log_std_max = _require_finite_float(
            self.policy_log_std_max,
            "policy_log_std_max",
        )
        if policy_log_std_min >= policy_log_std_max:
            raise ValueError("policy_log_std_min must be less than policy_log_std_max")
        object.__setattr__(self, "policy_log_std_min", policy_log_std_min)
        object.__setattr__(self, "policy_log_std_max", policy_log_std_max)
        object.__setattr__(
            self,
            "policy_head_gain",
            _require_positive_float(self.policy_head_gain, "policy_head_gain"),
        )
        object.__setattr__(
            self,
            "value_head_gain",
            _require_positive_float(self.value_head_gain, "value_head_gain"),
        )


class ActorCriticOutput(NamedTuple):
    action_mean: torch.Tensor
    action_log_std: torch.Tensor
    value: torch.Tensor


class PolicyAction(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    action_mean: torch.Tensor
    action_log_std: torch.Tensor


class PolicyEvaluation(NamedTuple):
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def _activation_module(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    *,
    activation: str = "tanh",
    layer_norm: bool = False,
) -> tuple[nn.Module, int]:
    input_dim = _require_positive_int(input_dim, "input_dim")
    hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
    if not isinstance(layer_norm, bool):
        raise TypeError("layer_norm must be bool")

    if not hidden_sizes:
        return nn.Identity(), input_dim

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(previous_dim, hidden_size))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(_activation_module(activation))
        previous_dim = hidden_size
    return nn.Sequential(*layers), previous_dim


def diagonal_gaussian_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    action = _require_float_tensor_2d(action, "action")
    mean = _require_float_tensor_2d(mean, "mean")
    log_std = _require_float_tensor_2d(log_std, "log_std")
    if action.shape != mean.shape or action.shape != log_std.shape:
        raise ValueError("action, mean, and log_std must have the same shape")

    var_inv = torch.exp(-2.0 * log_std)
    log_prob_per_dim = -0.5 * (
        (action - mean).pow(2) * var_inv + 2.0 * log_std + LOG_2PI
    )
    return log_prob_per_dim.sum(dim=-1)


def diagonal_gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    log_std = _require_float_tensor_2d(log_std, "log_std")
    return (log_std + 0.5 * (1.0 + LOG_2PI)).sum(dim=-1)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = EXECUTION_ACTION_DIM,
        *,
        config: ActorCriticConfig = ActorCriticConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(config, ActorCriticConfig):
            raise TypeError("config must be an ActorCriticConfig")

        self.obs_dim = _require_positive_int(obs_dim, "obs_dim")
        self.action_dim = _require_positive_int(action_dim, "action_dim")
        self.config = config

        self.backbone, features_dim = build_mlp(
            self.obs_dim,
            config.hidden_sizes,
            activation=config.activation,
            layer_norm=config.layer_norm,
        )

        self.action_mean_head = nn.Linear(features_dim, self.action_dim)
        self.value_head = nn.Linear(features_dim, 1)
        self.action_log_std = nn.Parameter(
            torch.full((self.action_dim,), config.policy_log_std_init)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.orthogonal_init:
            for module in self.backbone.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(
                self.action_mean_head.weight,
                gain=self.config.policy_head_gain,
            )
            nn.init.zeros_(self.action_mean_head.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=self.config.value_head_gain)
            nn.init.zeros_(self.value_head.bias)

        self.action_log_std.data.fill_(self.config.policy_log_std_init)

    def forward(self, obs: torch.Tensor) -> ActorCriticOutput:
        obs = _require_float_tensor_2d(obs, "obs")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError("obs last dimension must equal obs_dim")

        features = self.backbone(obs)
        action_mean = self.action_mean_head(features)
        clamped_log_std = torch.clamp(
            self.action_log_std,
            min=self.config.policy_log_std_min,
            max=self.config.policy_log_std_max,
        )
        action_log_std = clamped_log_std.expand_as(action_mean)
        value = self.value_head(features).squeeze(-1)
        return ActorCriticOutput(action_mean, action_log_std, value)

    def sample_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyAction:
        output = self.forward(obs)
        if deterministic:
            action = output.action_mean
        else:
            std = torch.exp(output.action_log_std)
            action = output.action_mean + torch.randn_like(output.action_mean) * std
        log_prob = diagonal_gaussian_log_prob(
            action,
            output.action_mean,
            output.action_log_std,
        )
        entropy = diagonal_gaussian_entropy(output.action_log_std)
        return PolicyAction(
            action,
            log_prob,
            entropy,
            output.value,
            output.action_mean,
            output.action_log_std,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyEvaluation:
        actions = _require_float_tensor_2d(actions, "actions")
        output = self.forward(obs)
        if actions.shape != output.action_mean.shape:
            raise ValueError("actions shape must match action_mean shape")
        log_prob = diagonal_gaussian_log_prob(
            actions,
            output.action_mean,
            output.action_log_std,
        )
        entropy = diagonal_gaussian_entropy(output.action_log_std)
        return PolicyEvaluation(log_prob, entropy, output.value)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )
