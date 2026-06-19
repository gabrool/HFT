from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn


# Flat hybrid action layout shared with ExecutionEnv:
# [bid_enable, ask_enable, bid_cancel_guard, ask_cancel_guard] Bernoulli flags
# followed by [bid_price_raw, ask_price_raw, bid_size_raw, ask_size_raw].
EXECUTION_ACTION_DIM = 8
EXECUTION_ENABLE_DIMS = 4
EXECUTION_CONTINUOUS_DIMS = EXECUTION_ACTION_DIM - EXECUTION_ENABLE_DIMS

ACTION_COMPONENT_NAMES = (
    "bid_enabled",
    "ask_enabled",
    "bid_price_raw",
    "ask_price_raw",
    "bid_size_raw",
    "ask_size_raw",
)

LOG_2PI = math.log(2.0 * math.pi)

__all__ = [
    "EXECUTION_ACTION_DIM",
    "EXECUTION_ENABLE_DIMS",
    "EXECUTION_CONTINUOUS_DIMS",
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

    enable_threshold: float = 0.5
    enable_logit_bias_init: float = 0.0
    continuous_log_std_init: float = -0.5
    continuous_log_std_min: float = -5.0
    continuous_log_std_max: float = 2.0

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

        threshold = _require_finite_float(self.enable_threshold, "enable_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("enable_threshold must be in [0, 1]")
        object.__setattr__(self, "enable_threshold", threshold)
        object.__setattr__(self, "enable_logit_bias_init", _require_finite_float(self.enable_logit_bias_init, "enable_logit_bias_init"))
        object.__setattr__(
            self,
            "continuous_log_std_init",
            _require_finite_float(self.continuous_log_std_init, "continuous_log_std_init"),
        )
        continuous_log_std_min = _require_finite_float(
            self.continuous_log_std_min,
            "continuous_log_std_min",
        )
        continuous_log_std_max = _require_finite_float(
            self.continuous_log_std_max,
            "continuous_log_std_max",
        )
        if continuous_log_std_min >= continuous_log_std_max:
            raise ValueError("continuous_log_std_min must be less than continuous_log_std_max")
        object.__setattr__(self, "continuous_log_std_min", continuous_log_std_min)
        object.__setattr__(self, "continuous_log_std_max", continuous_log_std_max)
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
    enable_logits: torch.Tensor
    continuous_mean: torch.Tensor
    continuous_log_std: torch.Tensor
    value: torch.Tensor


class PolicyAction(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    enable_prob: torch.Tensor
    enable_logits: torch.Tensor
    continuous_mean: torch.Tensor
    continuous_log_std: torch.Tensor


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


def _bernoulli_log_prob(action: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    action = _require_float_tensor_2d(action, "action")
    logits = _require_float_tensor_2d(logits, "logits")
    if action.shape != logits.shape:
        raise ValueError("action and logits must have the same shape")
    return -torch.nn.functional.binary_cross_entropy_with_logits(logits, action, reduction="none")


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = _require_float_tensor_2d(logits, "logits")
    probs = torch.sigmoid(logits)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction="none")


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
        if self.action_dim != EXECUTION_ACTION_DIM:
            raise ValueError("hybrid execution policy action_dim must be 8")
        self.config = config

        self.backbone, features_dim = build_mlp(
            self.obs_dim,
            config.hidden_sizes,
            activation=config.activation,
            layer_norm=config.layer_norm,
        )

        self.enable_head = nn.Linear(features_dim, EXECUTION_ENABLE_DIMS)
        self.continuous_mean_head = nn.Linear(features_dim, EXECUTION_CONTINUOUS_DIMS)
        self.value_head = nn.Linear(features_dim, 1)
        self.continuous_log_std = nn.Parameter(
            torch.full((EXECUTION_CONTINUOUS_DIMS,), config.continuous_log_std_init)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.orthogonal_init:
            for module in self.backbone.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(
                self.continuous_mean_head.weight,
                gain=self.config.policy_head_gain,
            )
            nn.init.zeros_(self.continuous_mean_head.bias)
            nn.init.orthogonal_(self.enable_head.weight, gain=self.config.policy_head_gain)
            nn.init.constant_(self.enable_head.bias, self.config.enable_logit_bias_init)
            nn.init.orthogonal_(self.value_head.weight, gain=self.config.value_head_gain)
            nn.init.zeros_(self.value_head.bias)

        self.continuous_log_std.data.fill_(self.config.continuous_log_std_init)
        nn.init.constant_(self.enable_head.bias, self.config.enable_logit_bias_init)

    def forward(self, obs: torch.Tensor) -> ActorCriticOutput:
        obs = _require_float_tensor_2d(obs, "obs")
        if obs.shape[-1] != self.obs_dim:
            raise ValueError("obs last dimension must equal obs_dim")

        features = self.backbone(obs)
        enable_logits = self.enable_head(features)
        continuous_mean = self.continuous_mean_head(features)
        clamped_log_std = torch.clamp(
            self.continuous_log_std,
            min=self.config.continuous_log_std_min,
            max=self.config.continuous_log_std_max,
        )
        continuous_log_std = clamped_log_std.expand_as(continuous_mean)
        value = self.value_head(features).squeeze(-1)
        return ActorCriticOutput(enable_logits, continuous_mean, continuous_log_std, value)

    def sample_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyAction:
        output = self.forward(obs)
        if deterministic:
            enable = (torch.sigmoid(output.enable_logits) >= self.config.enable_threshold).to(output.continuous_mean.dtype)
            continuous = output.continuous_mean
        else:
            enable = torch.bernoulli(torch.sigmoid(output.enable_logits))
            std = torch.exp(output.continuous_log_std)
            continuous = output.continuous_mean + torch.randn_like(output.continuous_mean) * std
        action = torch.cat((enable, continuous), dim=-1)
        log_prob = _bernoulli_log_prob(enable, output.enable_logits).sum(dim=-1) + diagonal_gaussian_log_prob(
            continuous, output.continuous_mean, output.continuous_log_std
        )
        entropy = _bernoulli_entropy(output.enable_logits).sum(dim=-1) + diagonal_gaussian_entropy(output.continuous_log_std)
        enable_prob = torch.sigmoid(output.enable_logits)
        return PolicyAction(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=output.value,
            enable_prob=enable_prob,
            enable_logits=output.enable_logits,
            continuous_mean=output.continuous_mean,
            continuous_log_std=output.continuous_log_std,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyEvaluation:
        actions = _require_float_tensor_2d(actions, "actions")
        output = self.forward(obs)
        if actions.shape != (obs.shape[0], self.action_dim):
            raise ValueError("actions shape must match policy action shape")
        enable = actions[:, :EXECUTION_ENABLE_DIMS]
        continuous = actions[:, EXECUTION_ENABLE_DIMS:]
        log_prob = _bernoulli_log_prob(enable, output.enable_logits).sum(dim=-1) + diagonal_gaussian_log_prob(
            continuous, output.continuous_mean, output.continuous_log_std
        )
        entropy = _bernoulli_entropy(output.enable_logits).sum(dim=-1) + diagonal_gaussian_entropy(output.continuous_log_std)
        return PolicyEvaluation(log_prob, entropy, output.value)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )
