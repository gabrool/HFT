import numpy as np

import pytest
import torch

from mmrt.rl.device import (
    canonicalize_torch_device,
    resolve_torch_device,
    torch_device_summary,
)
from mmrt.rl.rollout import RolloutCollector, RolloutConfig, _observation_to_tensor
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork
from tests.test_ppo_tiny_env import _tiny_env


def test_resolve_torch_device_auto_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")


def test_resolve_torch_device_auto_and_bare_cuda_are_indexed_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    expected = torch.device("cuda:2")
    assert resolve_torch_device("auto") == expected
    assert resolve_torch_device("cuda") == expected
    assert resolve_torch_device(torch.device("cuda")) == expected
    assert canonicalize_torch_device(torch.device("cuda")) == expected
    assert resolve_torch_device("cuda:0") == torch.device("cuda:0")


def test_torch_device_summary_reports_indexed_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Mock CUDA")

    resolved = resolve_torch_device("auto")
    summary = torch_device_summary(requested_device="auto", resolved_device=resolved)

    assert summary["resolved_device"] == "cuda:1"
    assert summary["cuda_available"] is True
    assert summary["cuda_device_name"] == "Mock CUDA"


def test_observation_to_tensor_out_validation_stays_strict_on_cpu():
    obs = np.array([1.0, 2.0], dtype=np.float32)
    valid_out = torch.empty((2,), device="cpu", dtype=torch.float32)

    result = _observation_to_tensor(
        obs,
        device=torch.device("cpu"),
        dtype=torch.float32,
        obs_dim=2,
        out=valid_out,
    )
    assert result is valid_out
    torch.testing.assert_close(valid_out, torch.tensor([1.0, 2.0], dtype=torch.float32))

    with pytest.raises(ValueError, match="out must match observation device, dtype, and shape"):
        _observation_to_tensor(
            obs,
            device=torch.device("cpu"),
            dtype=torch.float32,
            obs_dim=2,
            out=torch.empty((2,), device="cpu", dtype=torch.float64),
        )

    with pytest.raises(ValueError, match="out must match observation device, dtype, and shape"):
        _observation_to_tensor(
            obs,
            device=torch.device("cpu"),
            dtype=torch.float32,
            obs_dim=2,
            out=torch.empty((3,), device="cpu", dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="out must match observation device, dtype, and shape"):
        _observation_to_tensor(
            obs,
            device=torch.device("cpu"),
            dtype=torch.float32,
            obs_dim=2,
            out=torch.empty((2,), device="meta", dtype=torch.float32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_observation_to_tensor_accepts_bare_cuda_alias_for_indexed_out():
    device = torch.device("cuda")
    out = torch.empty((2,), device=resolve_torch_device("auto"), dtype=torch.float32)

    result = _observation_to_tensor(
        np.array([3.0, 4.0], dtype=np.float32),
        device=device,
        dtype=torch.float32,
        obs_dim=2,
        out=out,
    )

    assert result is out
    torch.testing.assert_close(out.detach().cpu(), torch.tensor([3.0, 4.0], dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rollout_collector_device_auto_resets_and_collects_on_cuda():
    env = _tiny_env()
    obs_dim = env.config.observation_schema.dim
    device = resolve_torch_device("auto")
    policy = ActorCriticNetwork(
        obs_dim=obs_dim,
        config=ActorCriticConfig(hidden_sizes=(8,)),
    ).to(device=device, dtype=torch.float32)
    collector = RolloutCollector(
        env,
        policy,
        config=RolloutConfig(
            rollout_steps=2,
            num_envs=1,
            device="auto",
            dtype=torch.float32,
        ),
    )

    reset_obs = collector.reset()
    batch = collector.collect()

    assert reset_obs.device == device
    assert batch.observations.device == device
    assert batch.num_steps == 2
