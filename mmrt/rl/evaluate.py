"""Policy evaluation loop for execution PPO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from mmrt.execution.diagnostics import ExecutionDiagnosticsConfig, diagnose_execution_metrics
from mmrt.execution.env import ExecutionEnv
from mmrt.execution.metrics import ExecutionMetricAccumulator
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.torch_networks import ActorCriticNetwork

__all__ = [
    "PolicyEvaluationConfig",
    "PolicyEvaluationResult",
    "evaluate_policy",
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


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative int or None")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative int or None")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


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
) -> torch.Tensor:
    obs_dim = _require_positive_int(obs_dim, "obs_dim")
    dtype = _require_float_dtype(dtype, "dtype")
    tensor = torch.as_tensor(obs, device=device, dtype=dtype)
    if tuple(tensor.shape) != (obs_dim,):
        raise ValueError(f"observation shape must be ({obs_dim},)")
    return tensor.clone()


@dataclass(frozen=True, slots=True)
class PolicyEvaluationConfig:
    max_steps: int | None = None
    start_event_index: int | None = None

    deterministic: bool = True
    reset_env: bool = True

    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32

    include_diagnostics: bool = True
    diagnostics_config: ExecutionDiagnosticsConfig = field(
        default_factory=ExecutionDiagnosticsConfig
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_steps",
            _optional_positive_int(self.max_steps, "max_steps"),
        )
        object.__setattr__(
            self,
            "start_event_index",
            _optional_nonnegative_int(self.start_event_index, "start_event_index"),
        )
        object.__setattr__(
            self,
            "deterministic",
            _require_bool(self.deterministic, "deterministic"),
        )
        object.__setattr__(
            self,
            "reset_env",
            _require_bool(self.reset_env, "reset_env"),
        )
        if self.device is not None and not isinstance(self.device, (str, torch.device)):
            raise TypeError("device must be None, str, or torch.device")
        object.__setattr__(self, "dtype", _require_float_dtype(self.dtype, "dtype"))
        object.__setattr__(
            self,
            "include_diagnostics",
            _require_bool(self.include_diagnostics, "include_diagnostics"),
        )
        if not isinstance(self.diagnostics_config, ExecutionDiagnosticsConfig):
            raise TypeError("diagnostics_config must be ExecutionDiagnosticsConfig")
        if not self.reset_env and self.start_event_index is not None:
            raise ValueError("start_event_index requires reset_env=True")


class PolicyEvaluationResult(NamedTuple):
    status: str
    steps: int
    terminated: bool
    truncated: bool
    metrics: dict[str, object]
    diagnostics: dict[str, object] | None
    config: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "steps": int(self.steps),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def _config_to_dict(config: PolicyEvaluationConfig) -> dict[str, object]:
    values: dict[str, object] = {
        "max_steps": config.max_steps,
        "start_event_index": config.start_event_index,
        "deterministic": config.deterministic,
        "reset_env": config.reset_env,
        "device": None if config.device is None else str(config.device),
        "dtype": str(config.dtype),
        "include_diagnostics": config.include_diagnostics,
    }
    as_dict = getattr(config.diagnostics_config, "as_dict", None)
    if callable(as_dict):
        values["diagnostics_config"] = as_dict()
    return values


def evaluate_policy(
    env: ExecutionEnv,
    policy: ActorCriticNetwork,
    *,
    config: PolicyEvaluationConfig = PolicyEvaluationConfig(),
    observation_normalizer: ObservationNormalizer | None = None,
) -> PolicyEvaluationResult:
    if not isinstance(env, ExecutionEnv):
        raise TypeError("env must be ExecutionEnv")
    if not isinstance(policy, ActorCriticNetwork):
        raise TypeError("policy must be ActorCriticNetwork")
    if not isinstance(config, PolicyEvaluationConfig):
        raise TypeError("config must be PolicyEvaluationConfig")
    if observation_normalizer is not None and not isinstance(
        observation_normalizer,
        ObservationNormalizer,
    ):
        raise TypeError("observation_normalizer must be None or ObservationNormalizer")
    if policy.obs_dim != env.config.observation_schema.dim:
        raise ValueError("policy.obs_dim must equal env observation dimension")
    if not config.reset_env and config.start_event_index is not None:
        raise ValueError("start_event_index requires reset_env=True")
    if not config.reset_env:
        raise NotImplementedError("reset_env=False is not supported yet")

    device = _resolve_device(policy, config.device)
    dtype = config.dtype
    policy_device = _resolve_device(policy, None)
    if device != policy_device:
        raise ValueError("evaluation device must match policy device")

    was_training = policy.training
    policy.eval()
    try:
        reset = env.reset(start_event_index=config.start_event_index)
        current_observation = reset.observation

        acc = ExecutionMetricAccumulator()
        terminated = False
        truncated = False
        step_count = 0

        while True:
            if config.max_steps is not None and step_count >= config.max_steps:
                truncated = True
                break

            obs = _observation_to_tensor(
                current_observation,
                device=device,
                dtype=dtype,
                obs_dim=policy.obs_dim,
            )
            if observation_normalizer is not None:
                obs = observation_normalizer.normalize(obs)

            with torch.no_grad():
                action_out = policy.sample_action(
                    obs.unsqueeze(0),
                    deterministic=config.deterministic,
                )
            action = action_out.action.squeeze(0).detach().to("cpu").tolist()

            step = env.step(action)
            acc.update(step.execution)
            step_count += 1

            current_observation = step.observation
            if step.done or step.truncated:
                terminated = bool(step.done)
                truncated = bool(step.truncated)
                break

        metrics = acc.as_dict()
        diagnostics = None
        status = "ok"
        if config.include_diagnostics:
            report = diagnose_execution_metrics(
                metrics,
                config=config.diagnostics_config,
            )
            diagnostics = report.as_dict()
            status = report.status

        return PolicyEvaluationResult(
            status=status,
            steps=step_count,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
            diagnostics=diagnostics,
            config=_config_to_dict(config),
        )
    finally:
        policy.train(was_training)
