"""Torch running-stat and normalization utilities for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


__all__ = [
    "RunningMeanStd",
    "ObservationNormalizerConfig",
    "ObservationNormalizer",
    "normalize_tensor",
    "normalize_advantages",
]


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
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


def _shape_tuple(shape: int | Sequence[int] | torch.Size, name: str) -> tuple[int, ...]:
    if isinstance(shape, bool):
        raise TypeError(f"{name} must be an int, sequence of ints, or torch.Size")
    if isinstance(shape, int):
        shape_tuple = (shape,)
    elif isinstance(shape, torch.Size):
        shape_tuple = tuple(shape)
    elif isinstance(shape, Sequence):
        shape_tuple = tuple(shape)
    else:
        raise TypeError(f"{name} must be an int, sequence of ints, or torch.Size")

    for dim in shape_tuple:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(f"{name} dimensions must be positive ints")
        if dim <= 0:
            raise ValueError(f"{name} dimensions must be positive ints")
    return shape_tuple


def _require_float_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating point tensor")
    return tensor


def normalize_tensor(
    values: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    clip: float | None = None,
) -> torch.Tensor:
    values = _require_float_tensor(values, "values")
    mean = _require_float_tensor(mean, "mean")
    var = _require_float_tensor(var, "var")
    epsilon = _require_positive_float(epsilon, "epsilon")
    if clip is not None:
        clip = _require_positive_float(clip, "clip")

    mean = mean.to(device=values.device, dtype=values.dtype)
    var = var.to(device=values.device, dtype=values.dtype)
    normalized = (values - mean) / torch.sqrt(var + epsilon)
    if clip is not None:
        normalized = torch.clamp(normalized, -clip, clip)
    return normalized


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        shape: int | Sequence[int] | torch.Size = (),
        *,
        epsilon: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        shape_tuple = _shape_tuple(shape, "shape")
        epsilon = _require_positive_float(epsilon, "epsilon")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=dtype).dtype.is_floating_point:
            raise TypeError("dtype must be a floating point dtype")

        self.shape = shape_tuple
        self.initial_epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape_tuple, dtype=dtype))
        self.register_buffer("var", torch.ones(shape_tuple, dtype=dtype))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=dtype))

    def reset(self) -> None:
        with torch.no_grad():
            self.mean.zero_()
            self.var.fill_(1.0)
            self.count.fill_(self.initial_epsilon)

    def _validate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = _require_float_tensor(batch, "batch")
        event_ndim = len(self.shape)

        if event_ndim == 0:
            if batch.ndim == 0:
                return batch.reshape(1)
            return batch.reshape(-1)

        if tuple(batch.shape[-event_ndim:]) != self.shape:
            if tuple(batch.shape) == self.shape:
                return batch.reshape((1,) + self.shape)
            raise ValueError(f"batch event shape must equal {self.shape}")

        return batch.reshape((-1,) + self.shape)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        batch = self._validate_batch(batch)
        if batch.shape[0] == 0:
            return

        batch = batch.to(device=self.mean.device, dtype=self.mean.dtype)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        self.update_from_moments(batch_mean, batch_var, batch.shape[0])

    @torch.no_grad()
    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int | float,
    ) -> None:
        batch_mean = _require_float_tensor(batch_mean, "batch_mean")
        batch_var = _require_float_tensor(batch_var, "batch_var")
        if tuple(batch_mean.shape) != self.shape:
            raise ValueError("batch_mean shape must match running stats shape")
        if tuple(batch_var.shape) != self.shape:
            raise ValueError("batch_var shape must match running stats shape")
        batch_count = _require_positive_float(batch_count, "batch_count")

        batch_mean = batch_mean.to(device=self.mean.device, dtype=self.mean.dtype)
        batch_var = batch_var.to(device=self.var.device, dtype=self.var.dtype)
        total_count = self.count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=0.0))
        self.count.copy_(total_count)

    def normalize(
        self,
        values: torch.Tensor,
        *,
        epsilon: float = 1e-8,
        clip: float | None = None,
    ) -> torch.Tensor:
        return normalize_tensor(values, self.mean, self.var, epsilon=epsilon, clip=clip)

    def denormalize(
        self,
        values: torch.Tensor,
        *,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        values = _require_float_tensor(values, "values")
        epsilon = _require_positive_float(epsilon, "epsilon")
        var = self.var.to(device=values.device, dtype=values.dtype)
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        return values * torch.sqrt(var + epsilon) + mean

    def extra_repr(self) -> str:
        return f"shape={self.shape}, count={float(self.count.item()):.1f}"


@dataclass(frozen=True, slots=True)
class ObservationNormalizerConfig:
    enabled: bool = True
    update: bool = True
    epsilon: float = 1e-8
    clip: float | None = 10.0
    rms_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "enabled"))
        object.__setattr__(self, "update", _require_bool(self.update, "update"))
        object.__setattr__(
            self,
            "epsilon",
            _require_positive_float(self.epsilon, "epsilon"),
        )
        if self.clip is not None:
            object.__setattr__(self, "clip", _require_positive_float(self.clip, "clip"))
        object.__setattr__(
            self,
            "rms_epsilon",
            _require_positive_float(self.rms_epsilon, "rms_epsilon"),
        )


class ObservationNormalizer(nn.Module):
    def __init__(
        self,
        obs_shape: int | Sequence[int] | torch.Size,
        *,
        config: ObservationNormalizerConfig = ObservationNormalizerConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(config, ObservationNormalizerConfig):
            raise TypeError("config must be an ObservationNormalizerConfig")

        self.obs_shape = _shape_tuple(obs_shape, "obs_shape")
        self.config = config
        self.running = RunningMeanStd(self.obs_shape, epsilon=config.rms_epsilon)

    def update(self, obs: torch.Tensor) -> None:
        if self.config.enabled:
            self.running.update(obs)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return obs
        return self.running.normalize(
            obs,
            epsilon=self.config.epsilon,
            clip=self.config.clip,
        )

    def update_and_normalize(
        self,
        obs: torch.Tensor,
        *,
        update: bool | None = None,
    ) -> torch.Tensor:
        if not self.config.enabled:
            return obs
        should_update = (
            self.config.update if update is None else _require_bool(update, "update")
        )
        if should_update:
            self.running.update(obs)
        return self.normalize(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalize(obs)

    def extra_repr(self) -> str:
        return (
            f"obs_shape={self.obs_shape}, "
            f"enabled={self.config.enabled}, "
            f"clip={self.config.clip}"
        )


def normalize_advantages(
    advantages: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    advantages = _require_float_tensor(advantages, "advantages")
    epsilon = _require_positive_float(epsilon, "epsilon")
    if advantages.numel() == 0:
        raise ValueError("advantages must not be empty")

    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    centered = advantages - mean
    if bool(std <= epsilon):
        return centered
    return centered / (std + epsilon)
