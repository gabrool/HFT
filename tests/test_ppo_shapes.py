import inspect
import math

import pytest
import torch

from mmrt.rl.normalization import (
    ObservationNormalizer,
    ObservationNormalizerConfig,
    RunningMeanStd,
    normalize_advantages,
)
from mmrt.rl import rollout as rollout_mod
from mmrt.rl.ppo import (
    PPOConfig,
    compute_ppo_loss,
    flatten_rollout_batch,
    iter_minibatch_indices,
    update_ppo,
)
from mmrt.rl.rollout import RolloutBatch, RolloutConfig, compute_discount_factors_from_dt_us, compute_gae
from mmrt.rl.torch_networks import (
    EXECUTION_ACTION_DIM,
    ActorCriticConfig,
    ActorCriticNetwork,
    diagonal_gaussian_entropy,
    diagonal_gaussian_log_prob,
)
from mmrt.rl.train import (
    PPOTrainingConfig,
    make_training_checkpoint_payload,
    training_config_to_dict,
)


def _synthetic_batch(
    *,
    steps: int = 8,
    obs_dim: int = 4,
    action_dim: int = EXECUTION_ACTION_DIM,
) -> RolloutBatch:
    return RolloutBatch(
        observations=torch.randn(steps, obs_dim),
        actions=torch.cat((torch.randint(0, 2, (steps, 2), dtype=torch.float32), torch.randn(steps, action_dim - 2)), dim=-1),
        log_probs=torch.zeros(steps),
        values=torch.zeros(steps),
        rewards=torch.randn(steps),
        dones=torch.zeros(steps, dtype=torch.bool),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
        advantages=torch.randn(steps),
        returns=torch.randn(steps),
        entropies=torch.zeros(steps),
        episode_count=0,
    )


def test_actor_critic_shapes_and_deterministic_action():
    policy = ActorCriticNetwork(obs_dim=4)
    obs = torch.zeros(3, 4)

    forward = policy(obs)
    assert forward.enable_logits.shape == (3, 4)
    assert forward.continuous_mean.shape == (3, 4)
    assert forward.continuous_log_std.shape == (3, 4)
    assert forward.value.shape == (3,)
    assert torch.isfinite(forward.enable_logits).all()
    assert torch.isfinite(forward.continuous_mean).all()
    assert torch.isfinite(forward.continuous_log_std).all()
    assert torch.isfinite(forward.value).all()

    sample = policy.sample_action(obs, deterministic=True)
    assert torch.all(sample.action[:, :4] == 1.0)
    assert torch.allclose(sample.action[:, 4:], forward.continuous_mean)
    assert sample.enable_prob.shape == (3, 4)
    assert sample.enable_logits.shape == (3, 4)
    assert sample.continuous_mean.shape == (3, 4)
    assert sample.continuous_log_std.shape == (3, 4)
    assert sample.log_prob.shape == (3,)
    assert sample.entropy.shape == (3,)
    assert torch.isfinite(sample.action).all()
    assert torch.isfinite(sample.log_prob).all()
    assert torch.isfinite(sample.entropy).all()

    evaluated = policy.evaluate_actions(obs, sample.action)
    assert evaluated.log_prob.shape == (3,)
    assert evaluated.entropy.shape == (3,)
    assert evaluated.value.shape == (3,)
    assert torch.isfinite(evaluated.log_prob).all()
    assert torch.isfinite(evaluated.entropy).all()
    assert torch.isfinite(evaluated.value).all()

    with pytest.raises(ValueError):
        policy(torch.zeros(4))
    with pytest.raises(ValueError):
        policy(torch.zeros(2, 5))
    with pytest.raises(ValueError):
        policy.evaluate_actions(torch.zeros(2, 4), torch.zeros(2, 5))


def test_actor_critic_log_std_clamping():
    config = ActorCriticConfig(
        continuous_log_std_init=10.0,
        continuous_log_std_min=-1.0,
        continuous_log_std_max=1.0,
    )
    policy = ActorCriticNetwork(obs_dim=4, config=config)
    out = policy(torch.zeros(2, 4))
    assert torch.allclose(out.continuous_log_std, torch.ones_like(out.continuous_log_std))

    with torch.no_grad():
        policy.continuous_log_std.fill_(-10.0)
    out = policy(torch.zeros(2, 4))
    assert torch.allclose(out.continuous_log_std, -torch.ones_like(out.continuous_log_std))


def test_diagonal_gaussian_helpers_shapes_and_values():
    action = torch.zeros(2, 3)
    mean = torch.zeros(2, 3)
    log_std = torch.zeros(2, 3)

    log_prob = diagonal_gaussian_log_prob(action, mean, log_std)
    entropy = diagonal_gaussian_entropy(log_std)

    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.allclose(log_prob, torch.full((2,), -0.5 * 3 * math.log(2.0 * math.pi)))
    assert torch.allclose(entropy, torch.full((2,), 3 * 0.5 * (1.0 + math.log(2.0 * math.pi))))

    with pytest.raises(ValueError):
        diagonal_gaussian_log_prob(action, mean[:, :2], log_std)
    with pytest.raises(ValueError):
        diagonal_gaussian_entropy(torch.zeros(3))


def test_running_mean_std_and_observation_normalizer():
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    rms = RunningMeanStd(shape=(3,))
    rms.update(x)
    z = rms.normalize(x)
    x_roundtrip = rms.denormalize(z)

    assert rms.mean.shape == (3,)
    assert rms.var.shape == (3,)
    assert z.shape == x.shape
    assert torch.allclose(x_roundtrip, x, atol=1e-5)

    norm = ObservationNormalizer(obs_shape=3)
    out = norm.update_and_normalize(x)
    assert out.shape == x.shape
    state = norm.state_dict()
    assert "running.mean" in state
    assert "running.var" in state
    assert "running.count" in state

    disabled = ObservationNormalizer(
        obs_shape=3,
        config=ObservationNormalizerConfig(enabled=False),
    )
    same = disabled.normalize(x)
    assert same is x


def test_normalize_advantages_standardizes_and_handles_constant():
    adv = torch.tensor([1.0, 2.0, 3.0])
    z = normalize_advantages(adv)
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert z.std(unbiased=False) == pytest.approx(1.0, abs=1e-6)

    constant = torch.ones(4)
    out = normalize_advantages(constant)
    assert torch.allclose(out, torch.zeros_like(out))

    with pytest.raises(ValueError):
        normalize_advantages(torch.tensor([]))


def test_compute_gae_shapes_and_terminal_mask():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([False, False, True])
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=torch.tensor(100.0),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.allclose(advantages, torch.tensor([2.5, 1.5, 0.5]))
    assert torch.allclose(returns, advantages + values)


def test_compute_gae_rank2_keeps_env_boundaries_independent():
    rewards = torch.tensor([[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[False, False], [True, False], [False, True]])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=torch.zeros(2),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages.shape == rewards.shape
    assert torch.allclose(advantages[:, 0], torch.tensor([2.0, 1.0, 1.0]))
    assert torch.allclose(advantages[:, 1], torch.tensor([30.0, 20.0, 10.0]))
    assert torch.allclose(returns, advantages)


def test_compute_gae_valid_mask_blocks_invalid_bootstrap_and_returns_values():
    rewards = torch.tensor([1.0, 10.0, 1.0])
    values = torch.tensor([0.0, 2.0, 0.0])
    dones = torch.tensor([False, False, False])
    valid_mask = torch.tensor([True, False, True])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=torch.tensor(0.0),
        gamma=1.0,
        gae_lambda=1.0,
        valid_mask=valid_mask,
    )

    assert torch.allclose(advantages, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(returns, torch.tensor([1.0, 2.0, 1.0]))


def _manual_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    discounts: torch.Tensor,
    lambda_discounts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.empty_like(rewards)
    next_gae = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = (~dones[t]).to(dtype=rewards.dtype)
        delta = rewards[t] + discounts[t] * next_value * nonterminal - values[t]
        next_gae = delta + discounts[t] * lambda_discounts[t] * nonterminal * next_gae
        advantages[t] = next_gae
    return advantages, advantages + values


def test_compute_gae_legacy_matches_scalar_formula():
    rewards = torch.tensor([1.0, 0.5, -0.25, 0.75])
    values = torch.tensor([0.1, 0.2, -0.1, 0.3])
    dones = torch.tensor([False, False, True, False])
    last_value = torch.tensor(0.4)
    gamma = 0.9
    gae_lambda = 0.8

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    expected_advantages, expected_returns = _manual_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        discounts=torch.full_like(rewards, gamma),
        lambda_discounts=torch.full_like(rewards, gae_lambda),
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_compute_gae_constant_time_step_matches_scalar_mode():
    rewards = torch.tensor([1.0, 0.5, 0.25])
    values = torch.tensor([0.1, 0.2, 0.3])
    dones = torch.tensor([False, False, False])
    last_value = torch.tensor(0.4)
    horizon_us = 1_000_000
    dt_us = torch.full_like(rewards, horizon_us, dtype=torch.int64)
    discounts = compute_discount_factors_from_dt_us(
        dt_us,
        factor_at_horizon=0.99,
        horizon_us=horizon_us,
    )
    lambda_discounts = compute_discount_factors_from_dt_us(
        dt_us,
        factor_at_horizon=0.95,
        horizon_us=horizon_us,
    )

    scalar_advantages, scalar_returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=0.99,
        gae_lambda=0.95,
    )
    time_advantages, time_returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=0.99,
        gae_lambda=0.95,
        discounts=discounts,
        lambda_discounts=lambda_discounts,
    )

    assert torch.allclose(discounts, torch.full_like(rewards, 0.99))
    assert torch.allclose(lambda_discounts, torch.full_like(rewards, 0.95))
    assert torch.allclose(time_advantages, scalar_advantages)
    assert torch.allclose(time_returns, scalar_returns)


def test_compute_gae_fractional_time_step_uses_sqrt_factors():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.1, 0.2, 0.3])
    dones = torch.tensor([False, False, False])
    last_value = torch.tensor(0.4)
    dt_us = torch.full_like(rewards, 50, dtype=torch.int64)
    discounts = compute_discount_factors_from_dt_us(dt_us, factor_at_horizon=0.81, horizon_us=100)
    lambda_discounts = compute_discount_factors_from_dt_us(dt_us, factor_at_horizon=0.64, horizon_us=100)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=0.81,
        gae_lambda=0.64,
        discounts=discounts,
        lambda_discounts=lambda_discounts,
    )

    assert torch.allclose(discounts, torch.full_like(rewards, 0.9))
    assert torch.allclose(lambda_discounts, torch.full_like(rewards, 0.8))
    assert torch.allclose(advantages, torch.tensor([4.156704, 4.2732, 3.06]))
    assert torch.allclose(returns, torch.tensor([4.256704, 4.4732, 3.36]))


def test_discount_factors_handle_zero_and_variable_dt():
    zero_dt = torch.tensor([0, 1], dtype=torch.int64)
    zero_factor_discounts = compute_discount_factors_from_dt_us(
        zero_dt,
        factor_at_horizon=0.0,
        horizon_us=100,
    )
    assert torch.equal(zero_factor_discounts, torch.tensor([1.0, 0.0]))
    assert torch.isfinite(zero_factor_discounts).all()

    dt_us = torch.tensor([0, 50, 100, 200], dtype=torch.int64)
    discounts = compute_discount_factors_from_dt_us(dt_us, factor_at_horizon=0.81, horizon_us=100)
    assert torch.isfinite(discounts).all()
    assert ((discounts >= 0.0) & (discounts <= 1.0)).all()
    assert torch.all(discounts[1:] <= discounts[:-1])


def test_rollout_batch_to_moves_discount_tensors():
    batch = _synthetic_batch(steps=3)
    with_discounts = batch._replace(
        step_dt_us=torch.tensor([0, 50, 100], dtype=torch.int64),
        discounts=torch.tensor([1.0, 0.9, 0.81]),
        lambda_discounts=torch.tensor([1.0, 0.8, 0.64]),
    )

    moved = with_discounts.to("cpu")

    assert moved.step_dt_us is not None
    assert moved.discounts is not None
    assert moved.lambda_discounts is not None
    assert moved.step_dt_us.device.type == "cpu"
    assert moved.discounts.device.type == "cpu"
    assert moved.lambda_discounts.device.type == "cpu"


def test_rollout_batch_to_moves_reward_projection_tensors():
    batch = _synthetic_batch(steps=3)._replace(
        env_rewards=torch.tensor([1.0, 2.0, 3.0]),
        projected_rewards=torch.tensor([1.5, 2.5, 3.5]),
        reward_valid_mask=torch.tensor([True, False, True]),
        reward_components={"path_equity_delta_H": torch.tensor([1.0, 0.0, 3.0])},
        reward_mode="horizon_path_equity",
        reward_projection_stats={"valid_fraction": 2.0 / 3.0},
    )

    moved = batch.to("cpu")

    assert moved.env_rewards is not None
    assert moved.projected_rewards is not None
    assert moved.reward_valid_mask is not None
    assert moved.reward_components is not None
    assert moved.env_rewards.device.type == "cpu"
    assert moved.projected_rewards.device.type == "cpu"
    assert moved.reward_valid_mask.device.type == "cpu"
    assert moved.reward_components["path_equity_delta_H"].device.type == "cpu"
    assert moved.reward_mode == "horizon_path_equity"
    assert moved.reward_projection_stats == {"valid_fraction": 2.0 / 3.0}


def test_flatten_rollout_batch_flattens_vectorized_rollout_at_update_time():
    steps = 3
    num_envs = 2
    obs_dim = 4
    action_dim = EXECUTION_ACTION_DIM
    batch = RolloutBatch(
        observations=torch.randn(steps, num_envs, obs_dim),
        actions=torch.randn(steps, num_envs, action_dim),
        log_probs=torch.zeros(steps, num_envs),
        values=torch.zeros(steps, num_envs),
        rewards=torch.randn(steps, num_envs),
        dones=torch.zeros(steps, num_envs, dtype=torch.bool),
        terminated=torch.zeros(steps, num_envs, dtype=torch.bool),
        truncated=torch.zeros(steps, num_envs, dtype=torch.bool),
        advantages=torch.randn(steps, num_envs),
        returns=torch.randn(steps, num_envs),
        entropies=torch.zeros(steps, num_envs),
        episode_count=0,
    )

    flat = flatten_rollout_batch(batch)

    assert batch.num_steps == steps * num_envs
    assert flat["observations"].shape == (steps * num_envs, obs_dim)
    assert flat["actions"].shape == (steps * num_envs, action_dim)
    assert flat["advantages"].shape == (steps * num_envs,)


def test_flatten_rollout_batch_filters_invalid_reward_anchors():
    batch = _synthetic_batch(steps=5)._replace(
        advantages=torch.arange(5, dtype=torch.float32),
        returns=torch.arange(10, 15, dtype=torch.float32),
        reward_valid_mask=torch.tensor([True, False, True, False, True]),
    )

    flat = flatten_rollout_batch(batch)

    assert flat["observations"].shape[0] == 3
    assert torch.equal(flat["advantages"], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(flat["returns"], torch.tensor([10.0, 12.0, 14.0]))


def test_flatten_rollout_batch_rejects_all_invalid_reward_anchors():
    batch = _synthetic_batch(steps=3)._replace(
        reward_valid_mask=torch.zeros(3, dtype=torch.bool),
    )

    with pytest.raises(RuntimeError, match="zero valid reward anchors"):
        flatten_rollout_batch(batch)


def test_observation_to_tensor_can_reuse_output_buffer():
    out = torch.empty(3, dtype=torch.float32)
    result = rollout_mod._observation_to_tensor(
        [1.0, 2.0, 3.0],
        device=torch.device("cpu"),
        dtype=torch.float32,
        obs_dim=3,
        out=out,
    )

    assert result is out
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))


def test_ppo_loss_and_update_shapes_and_stats():
    policy = ActorCriticNetwork(obs_dim=4, config=ActorCriticConfig(hidden_sizes=(8,)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch = _synthetic_batch(steps=8, obs_dim=4)
    config = PPOConfig(update_epochs=2, minibatch_size=4)

    flat = flatten_rollout_batch(batch)
    loss = compute_ppo_loss(policy, config=config, **flat)
    for tensor in loss:
        assert tensor.ndim == 0
        assert torch.isfinite(tensor)

    stats = update_ppo(policy, optimizer, batch, config=config)
    assert stats.minibatches_processed == 4
    assert stats.epochs_completed == 2
    assert stats.early_stop is False
    assert isinstance(stats.as_dict(), dict)
    assert all(isinstance(value, (bool, float, int)) for value in stats.as_dict().values())

    early = update_ppo(
        policy,
        optimizer,
        batch,
        config=PPOConfig(update_epochs=4, minibatch_size=4, target_kl=1e-12),
    )
    assert early.minibatches_processed >= 1
    assert early.early_stop in (True, False)


def test_iter_minibatch_indices_covers_all_rows():
    chunks = list(iter_minibatch_indices(10, 4, device=torch.device("cpu"), shuffle=False))
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert torch.equal(torch.cat(chunks), torch.arange(10))


def test_training_config_to_dict_is_json_safe():
    cfg = PPOTrainingConfig(
        num_updates=1,
        rollout_config=RolloutConfig(rollout_steps=4, num_envs=2, device="cpu"),
        ppo_config=PPOConfig(minibatch_size=4),
    )
    payload = training_config_to_dict(cfg)
    assert payload["num_updates"] == 1
    assert payload["rollout_config"]["device"] == "cpu"
    assert payload["rollout_config"]["num_envs"] == 2
    assert payload["rollout_config"]["effective_batch_size"] == 8
    assert payload["rollout_config"]["dtype"].startswith("torch.")
    net = payload["network_config"]
    assert "enable_threshold" in net
    assert "enable_logit_bias_init" in net
    assert "continuous_log_std_init" in net
    assert "policy" + "_log_std_init" not in net


def test_rl_modules_do_not_import_forbidden_heavy_or_wrong_layers():
    import mmrt.rl.normalization as normalization
    import mmrt.rl.ppo as ppo
    import mmrt.rl.torch_networks as torch_networks
    import mmrt.rl.train as train

    forbidden = (
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.storage",
        "mmrt.linear",
        "load_execution_tape",
    )
    for module in (torch_networks, normalization, ppo):
        source = inspect.getsource(module)
        for text in forbidden:
            assert text not in source

    ppo_source = inspect.getsource(ppo)
    assert "ExecutionEnv" not in ppo_source
    assert "ExecutionTape" not in ppo_source

    train_source = inspect.getsource(train)
    for text in (
        "load_execution_tape",
        "argparse",
        "torch.save",
        "torch.load",
        "pandas",
        "polars",
        "pyarrow",
        "gym",
        "gymnasium",
    ):
        assert text not in train_source
