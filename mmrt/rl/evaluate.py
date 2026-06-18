"""Policy evaluation loop for execution PPO policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from mmrt.execution.diagnostics import ExecutionDiagnosticsConfig, diagnose_execution_metrics
from mmrt.execution.env import ExecutionEnv
from mmrt.execution.metrics import ExecutionMetricAccumulator
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.rl.device import canonicalize_torch_device, resolve_torch_device
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
        return resolve_torch_device(device)

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
    start_decision_row: int | None = None
    end_decision_row: int | None = None
    decision_row_ranges: tuple[DecisionSplitRange, ...] = ()

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
            "start_decision_row",
            _optional_nonnegative_int(self.start_decision_row, "start_decision_row"),
        )
        object.__setattr__(
            self,
            "end_decision_row",
            _optional_nonnegative_int(self.end_decision_row, "end_decision_row"),
        )
        if self.start_decision_row is not None and self.end_decision_row is not None:
            if self.end_decision_row <= self.start_decision_row:
                raise ValueError("end_decision_row must be greater than start_decision_row")
        ranges = tuple(self.decision_row_ranges)
        if ranges:
            if self.start_decision_row is not None or self.end_decision_row is not None:
                raise ValueError("decision_row_ranges cannot be combined with start/end_decision_row")
            if any(not isinstance(item, DecisionSplitRange) for item in ranges):
                raise TypeError("decision_row_ranges entries must be DecisionSplitRange")
            if any(item.rollout_step_capacity <= 0 for item in ranges):
                raise ValueError("decision_row_ranges must each contain at least two decision rows")
        object.__setattr__(self, "decision_row_ranges", ranges)
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
        if not self.reset_env and self.start_decision_row is not None:
            raise ValueError("start_decision_row requires reset_env=True")


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
        "start_decision_row": config.start_decision_row,
        "end_decision_row": config.end_decision_row,
        "decision_row_ranges": [entry.as_dict() for entry in config.decision_row_ranges],
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


def _increment_counter(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


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
    if not config.reset_env and config.start_decision_row is not None:
        raise ValueError("start_decision_row requires reset_env=True")
    if not config.reset_env:
        raise NotImplementedError("reset_env=False is not supported yet")

    device = _resolve_device(policy, config.device)
    dtype = config.dtype
    policy_device = _resolve_device(policy, None)
    if canonicalize_torch_device(device) != canonicalize_torch_device(policy_device):
        raise ValueError("evaluation device must match policy device")

    was_training = policy.training
    policy.eval()
    try:
        acc = ExecutionMetricAccumulator()
        terminated = False
        truncated = False
        step_count = 0
        episode_count = 0
        evaluated_start_decision_row: int | None = None
        evaluated_end_decision_row: int | None = None
        evaluated_ranges: list[dict[str, object]] = []
        truncation_counts: dict[str, int] = {
            "max_episode_steps": 0,
            "split_exhausted": 0,
            "max_steps": 0,
            "terminal_done": 0,
        }

        if config.decision_row_ranges:
            eval_ranges = tuple(
                (entry.start_decision_row, entry.end_decision_row, entry.as_dict())
                for entry in config.decision_row_ranges
            )
        else:
            reset_start = 0 if config.start_decision_row is None else config.start_decision_row
            reset_end = config.end_decision_row
            eval_ranges = ((reset_start, reset_end, None),)

        requested_row_count = int(
            sum(
                max(0, int(range_end) - int(range_start) - 1)
                for range_start, range_end, _range_meta in eval_ranges
                if range_end is not None
            )
        )
        stop_all_ranges = False
        for range_start, range_end, range_meta in eval_ranges:
            cursor = int(range_start)
            while True:
                if config.max_steps is not None and step_count >= config.max_steps:
                    truncated = True
                    _increment_counter(truncation_counts, "max_steps")
                    stop_all_ranges = True
                    break
                if range_end is not None and cursor + 1 >= range_end:
                    truncated = True
                    _increment_counter(truncation_counts, "split_exhausted")
                    break

                reset = env.reset(start_decision_row=cursor)
                current_observation = reset.observation
                current_decision_row = int(reset.info["decision_grid_row_index"])
                if current_decision_row != cursor:
                    raise RuntimeError("evaluation reset did not land on requested decision row")
                if range_end is not None and current_decision_row + 1 >= range_end:
                    truncated = True
                    _increment_counter(truncation_counts, "split_exhausted")
                    break
                if evaluated_start_decision_row is None:
                    evaluated_start_decision_row = current_decision_row
                episode_count += 1
                active_range_start = current_decision_row
                active_range_end = current_decision_row

                while True:
                    if config.max_steps is not None and step_count >= config.max_steps:
                        truncated = True
                        _increment_counter(truncation_counts, "max_steps")
                        stop_all_ranges = True
                        break
                    if range_end is not None and current_decision_row + 1 >= range_end:
                        truncated = True
                        _increment_counter(truncation_counts, "split_exhausted")
                        cursor = current_decision_row
                        break

                    obs = _observation_to_tensor(
                        current_observation,
                        device=device,
                        dtype=dtype,
                        obs_dim=policy.obs_dim,
                    )
                    if observation_normalizer is not None:
                        obs = observation_normalizer.normalize(obs)

                    with torch.inference_mode():
                        action_out = policy.sample_action(
                            obs.unsqueeze(0),
                            deterministic=config.deterministic,
                        )
                    action = action_out.action.squeeze(0).detach().to("cpu").tolist()

                    action_decision_row = current_decision_row
                    step = env.step(action)
                    acc.update(step.execution)
                    step_count += 1
                    next_decision_row = int(step.info["next_decision_grid_row_index"])
                    if range_end is not None and next_decision_row >= range_end:
                        raise RuntimeError("evaluation step crossed split boundary")
                    active_range_end = action_decision_row + 1
                    evaluated_end_decision_row = active_range_end

                    current_observation = step.observation
                    current_decision_row = next_decision_row
                    cursor = next_decision_row
                    if step.done:
                        terminated = True
                        _increment_counter(truncation_counts, "terminal_done")
                        stop_all_ranges = True
                        break
                    if step.truncated:
                        truncated = True
                        _increment_counter(truncation_counts, "max_episode_steps")
                        break
                    if range_end is not None and current_decision_row + 1 >= range_end:
                        truncated = True
                        _increment_counter(truncation_counts, "split_exhausted")
                        break

                if active_range_end > active_range_start:
                    evaluated_range = {
                        "start_decision_row": int(active_range_start),
                        "end_decision_row": int(active_range_end),
                        "row_count": int(active_range_end - active_range_start),
                    }
                    if range_meta is not None:
                        evaluated_range = {**range_meta, **evaluated_range}
                    evaluated_ranges.append(evaluated_range)
                if stop_all_ranges:
                    break
                if range_end is None:
                    stop_all_ranges = True
                    break
                if cursor + 1 >= range_end:
                    break
            if stop_all_ranges:
                break

        covered_row_count = int(sum(int(entry["row_count"]) for entry in evaluated_ranges))
        coverage_fraction = (
            float(covered_row_count / requested_row_count)
            if requested_row_count > 0
            else 0.0
        )
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
            config={
                **_config_to_dict(config),
                "evaluated_start_decision_row": None if evaluated_start_decision_row is None else int(evaluated_start_decision_row),
                "evaluated_end_decision_row": None if evaluated_end_decision_row is None else int(evaluated_end_decision_row),
                "evaluated_decision_row_ranges": evaluated_ranges,
                "eval_requested_row_count": requested_row_count,
                "eval_covered_row_count": covered_row_count,
                "eval_coverage_fraction": coverage_fraction,
                "episode_count": int(episode_count),
                "truncation_counts": truncation_counts,
            },
        )
    finally:
        policy.train(was_training)
