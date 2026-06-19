from pathlib import Path

import torch

from mmrt.rl import ppo
from mmrt.rl.ppo import PPOConfig, update_ppo
from mmrt.rl.rollout import RolloutBatch
from mmrt.rl.torch_networks import EXECUTION_ACTION_DIM, ActorCriticConfig, ActorCriticNetwork


def _synthetic_batch(*, steps: int = 8, obs_dim: int = 4, action_dim: int = EXECUTION_ACTION_DIM) -> RolloutBatch:
    return RolloutBatch(
        observations=torch.randn(steps, obs_dim),
        actions=torch.cat((torch.randint(0, 2, (steps, 2), dtype=torch.float32), torch.randn(steps, action_dim - 2)), dim=-1),
        log_probs=torch.zeros(steps),
        values=torch.zeros(steps),
        rewards=torch.randn(steps),
        dones=torch.zeros(steps, dtype=torch.bool),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
        advantages=torch.arange(steps, dtype=torch.float32),
        returns=torch.randn(steps),
        entropies=torch.zeros(steps),
        episode_count=0,
    )


def test_update_ppo_normalizes_advantages_once_over_flat_batch(monkeypatch):
    calls = {"count": 0}
    original = ppo.normalize_advantages

    def counted(advantages):
        calls["count"] += 1
        return original(advantages)

    monkeypatch.setattr(ppo, "normalize_advantages", counted)
    policy = ActorCriticNetwork(obs_dim=4, config=ActorCriticConfig(hidden_sizes=(8,)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    update_ppo(policy, optimizer, _synthetic_batch(), config=PPOConfig(update_epochs=3, minibatch_size=2))
    assert calls["count"] == 1


def test_update_ppo_normalizes_advantages_only_over_valid_reward_anchors(monkeypatch):
    seen = []

    def counted(advantages):
        seen.append(advantages.detach().clone())
        return advantages

    monkeypatch.setattr(ppo, "normalize_advantages", counted)
    policy = ActorCriticNetwork(obs_dim=4, config=ActorCriticConfig(hidden_sizes=(8,)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch = _synthetic_batch(steps=5)._replace(
        advantages=torch.arange(5, dtype=torch.float32),
        reward_valid_mask=torch.tensor([True, False, True, False, True]),
    )

    update_ppo(policy, optimizer, batch, config=PPOConfig(update_epochs=1, minibatch_size=3))

    assert len(seen) == 1
    assert torch.equal(seen[0], torch.tensor([0.0, 2.0, 4.0]))


def test_update_ppo_honors_normalize_advantages_false(monkeypatch):
    def forbidden(_advantages):
        raise AssertionError("normalize_advantages should not be called")

    monkeypatch.setattr(ppo, "normalize_advantages", forbidden)
    policy = ActorCriticNetwork(obs_dim=4, config=ActorCriticConfig(hidden_sizes=(8,)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    update_ppo(policy, optimizer, _synthetic_batch(), config=PPOConfig(update_epochs=1, minibatch_size=4, normalize_advantages=False))


def test_compute_ppo_loss_does_not_normalize_advantages_per_minibatch():
    source = Path("mmrt/rl/ppo.py").read_text()
    body = source.split("def compute_ppo_loss", 1)[1].split("def update_ppo", 1)[0]
    assert "normalize_advantages(" not in body
