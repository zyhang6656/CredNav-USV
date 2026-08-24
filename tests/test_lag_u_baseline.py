from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces

from scripts.train_lag_u_baseline import env_config_for_lag_u, validate_training_config
from scripts.eval_lag_u_baseline import load_model, select_action

from simple_boat.envs.lag_u_baseline import (
    Actor,
    CostCritic,
    LagUAgent,
    LagUReplayBuffer,
    ReplayBatch,
    RewardCriticEnsemble,
    TrustFreeObservationWrapper,
    adaptive_cost_bound,
    actor_objective,
    discounted_target,
    ensemble_statistics,
    exploration_probability,
    flatten_trust_free_observation,
    transition_cost,
    uncertainty_ratio,
)


def test_flatten_trust_free_observation_removes_each_obstacle_trust() -> None:
    state = np.arange(13, dtype=np.float32)
    dyn = np.arange(42, dtype=np.float32).reshape(6, 7)
    mask = np.ones(6, dtype=np.float32)

    flat = flatten_trust_free_observation({"state": state, "dyn": dyn.reshape(-1), "dyn_mask": mask})

    assert flat.shape == (55,)
    np.testing.assert_array_equal(flat[:13], state)
    np.testing.assert_array_equal(flat[13:49], dyn[:, :6].reshape(-1))
    np.testing.assert_array_equal(flat[49:], mask)


def test_exploration_probability_matches_paper_schedule() -> None:
    assert exploration_probability(0, tc=300_000, minimum=0.1) == pytest.approx(1.0)
    assert exploration_probability(150_000, tc=300_000, minimum=0.1) == pytest.approx(0.5)
    assert exploration_probability(300_000, tc=300_000, minimum=0.1) == pytest.approx(0.1)
    assert exploration_probability(3_000_000, tc=300_000, minimum=0.1) == pytest.approx(0.1)


def test_ensemble_statistics_uses_population_variance() -> None:
    values = torch.tensor([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]])

    mean, variance = ensemble_statistics(values)

    torch.testing.assert_close(mean, torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(variance, torch.tensor([8.0 / 3.0, 8.0 / 3.0]))


def test_uncertainty_ratio_is_nonnegative_and_scale_invariant_for_signed_q() -> None:
    variance = torch.tensor([[1.0]])

    positive = uncertainty_ratio(torch.tensor([[2.0]]), variance)
    negative = uncertainty_ratio(torch.tensor([[-2.0]]), variance)
    scaled_negative = uncertainty_ratio(torch.tensor([[-20.0]]), variance * 100.0)

    torch.testing.assert_close(positive, torch.tensor([[0.5]]))
    torch.testing.assert_close(negative, positive)
    torch.testing.assert_close(scaled_negative, positive)


def test_adaptive_cost_bound_matches_paper_threshold_rule() -> None:
    ratio = torch.tensor([[0.03], [0.07], [0.14]])

    bound = adaptive_cost_bound(ratio, base_bound=0.1, threshold=0.07)

    torch.testing.assert_close(bound, torch.tensor([[0.1], [0.1], [0.05]]))


@pytest.mark.parametrize("reason", ["dynamic_obs", "static_obs", "out_of_bounds", "collision"])
def test_transition_cost_is_one_only_for_collision_events(reason: str) -> None:
    assert transition_cost({"reason": reason}) == 1.0


@pytest.mark.parametrize("info", [{}, {"reason": "goal_reached"}, {"timeout_reason": "max_steps"}])
def test_transition_cost_ignores_success_and_timeout(info: dict[str, str]) -> None:
    assert transition_cost(info) == 0.0


def test_lag_u_network_shapes_and_independent_ensemble_members() -> None:
    torch.manual_seed(7)
    actor = Actor(obs_dim=55, action_dim=2, hidden_dims=(256, 256))
    reward = RewardCriticEnsemble(obs_dim=55, action_dim=2, members=3, hidden_dims=(256, 256))
    cost = CostCritic(obs_dim=55, action_dim=2, hidden_dims=(256, 256))
    obs = torch.linspace(-1.0, 1.0, 55).repeat(4, 1)
    action = actor(obs)

    q1, q2 = reward(obs, action)

    assert action.shape == (4, 2)
    assert torch.all(action <= 1.0) and torch.all(action >= -1.0)
    assert q1.shape == (3, 4, 1)
    assert q2.shape == (3, 4, 1)
    assert cost(obs, action).shape == (4, 1)
    assert not torch.equal(q1[0], q1[1])


def test_cost_critic_outputs_unbounded_discounted_cost() -> None:
    cost = CostCritic(obs_dim=3, action_dim=2, hidden_dims=(4,))
    for parameter in cost.parameters():
        parameter.data.zero_()
    cost.network[-1].bias.data.fill_(5.0)

    values = cost(torch.zeros(2, 3), torch.zeros(2, 2))

    torch.testing.assert_close(values, torch.full((2, 1), 5.0))

    cost.network[-1].bias.data.fill_(-2.0)
    torch.testing.assert_close(
        cost(torch.zeros(2, 3), torch.zeros(2, 2)),
        torch.full((2, 1), -2.0),
    )


def test_policy_ensemble_uses_each_members_primary_critic_not_twin_minimum() -> None:
    reward = RewardCriticEnsemble(obs_dim=3, action_dim=2, members=3, hidden_dims=(4,))
    for index, member in enumerate(reward.members):
        for parameter in member.parameters():
            parameter.data.zero_()
        member.q1[-1].bias.data.fill_(float(index + 1))
        member.q2[-1].bias.data.fill_(-100.0)

    values = reward.policy_values(torch.zeros(2, 3), torch.zeros(2, 2))

    torch.testing.assert_close(values[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_discounted_target_stops_bootstrap_at_terminal_transition() -> None:
    immediate = torch.tensor([[1.0], [1.0]])
    next_value = torch.tensor([[10.0], [10.0]])
    done = torch.tensor([[0.0], [1.0]])

    target = discounted_target(immediate, next_value, done, gamma=0.9)

    torch.testing.assert_close(target, torch.tensor([[10.0], [1.0]]))


def test_actor_objective_matches_exploratory_and_safe_equations() -> None:
    member_values = torch.tensor([[[1.0]], [[3.0]], [[5.0]]])
    cost_values = torch.tensor([[0.2]])
    lagrange = torch.tensor(2.0)

    exploratory_loss, exploratory_bound, ratio = actor_objective(
        member_values,
        cost_values,
        lagrange,
        exploratory=True,
        base_bound=0.1,
        uncertainty_threshold=0.07,
    )
    safe_loss, safe_bound, _ = actor_objective(
        member_values,
        cost_values,
        lagrange,
        exploratory=False,
        base_bound=0.1,
        uncertainty_threshold=0.07,
    )

    expected_std = torch.tensor(8.0 / 3.0).sqrt()
    torch.testing.assert_close(exploratory_bound, torch.tensor([[0.1]]))
    torch.testing.assert_close(exploratory_loss, -torch.tensor(3.0) - expected_std + torch.tensor(0.2))
    torch.testing.assert_close(ratio, expected_std.reshape(1, 1) / 3.0)
    torch.testing.assert_close(safe_bound, torch.tensor([[0.1 * 0.07 / float(ratio.item())]]))
    torch.testing.assert_close(safe_loss, -torch.tensor(3.0) + 2.0 * (0.2 - safe_bound.mean()))


@pytest.mark.parametrize("exploratory", [True, False])
def test_actor_objective_has_finite_gradient_when_ensemble_agrees(exploratory: bool) -> None:
    member_values = torch.ones((3, 2, 1), requires_grad=True)
    loss, _, ratio = actor_objective(
        member_values,
        torch.zeros((2, 1)),
        torch.tensor(0.0),
        exploratory=exploratory,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(ratio).all()
    assert member_values.grad is not None
    assert torch.isfinite(member_values.grad).all()


def test_agent_delays_actor_update_and_keeps_lagrange_nonnegative() -> None:
    torch.manual_seed(11)
    agent = LagUAgent(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(16, 16),
        ensemble_members=3,
        actor_lr=3e-4,
        critic_lr=3e-4,
        lagrange_lr=1e-5,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.0,
        target_noise_clip=0.0,
        device="cpu",
    )
    batch = ReplayBatch(
        obs=torch.randn(8, 5),
        actions=torch.empty(8, 2).uniform_(-1.0, 1.0),
        rewards=torch.randn(8, 1),
        costs=torch.zeros(8, 1),
        next_obs=torch.randn(8, 5),
        dones=torch.zeros(8, 1),
    )
    actor_before = [value.detach().clone() for value in agent.actor.parameters()]

    first = agent.update(batch, environment_step=1, exploratory=False)
    actor_after_first = [value.detach().clone() for value in agent.actor.parameters()]
    second = agent.update(batch, environment_step=2, exploratory=False)
    actor_after_second = [value.detach().clone() for value in agent.actor.parameters()]

    assert first["actor_updated"] is False
    assert second["actor_updated"] is True
    assert all(torch.equal(before, after) for before, after in zip(actor_before, actor_after_first))
    assert any(not torch.equal(before, after) for before, after in zip(actor_after_first, actor_after_second))
    assert agent.lagrange.item() >= 0.0
    for key in ("q_std_mean", "uncertainty_ratio_p95", "cost_q_mean", "cost_bound_mean"):
        assert np.isfinite(second[key])
    assert np.isfinite(second["cost_q_mean"])
    assert 0.0 <= second["cost_bound_mean"] <= 0.1


def test_cost_target_uses_clean_target_action_without_td3_smoothing_noise() -> None:
    class RecordingCostCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_action: torch.Tensor | None = None

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            self.last_action = action.detach().clone()
            return torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)

    torch.manual_seed(19)
    agent = LagUAgent(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(8, 8),
        policy_delay=2,
        target_policy_noise=1.0,
        target_noise_clip=0.5,
        device="cpu",
    )
    recorder = RecordingCostCritic()
    agent.target_cost_critic = recorder
    batch = ReplayBatch(
        obs=torch.randn(4, 5),
        actions=torch.zeros(4, 2),
        rewards=torch.zeros(4, 1),
        costs=torch.zeros(4, 1),
        next_obs=torch.randn(4, 5),
        dones=torch.zeros(4, 1),
    )
    with torch.no_grad():
        expected = agent.target_actor(batch.next_obs)

    agent.update(batch, environment_step=1, exploratory=False)

    assert recorder.last_action is not None
    torch.testing.assert_close(recorder.last_action, expected)


def test_cost_target_uses_paper_discount_gamma() -> None:
    class ConstantCostCritic(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(float(value)))

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.value.expand(obs.shape[0], 1)

    agent = LagUAgent(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(8, 8),
        gamma=0.25,
        cost_gamma=0.99,
        policy_delay=2,
        device="cpu",
    )
    agent.cost_critic = ConstantCostCritic(0.0)
    agent.target_cost_critic = ConstantCostCritic(0.8)
    batch = ReplayBatch(
        obs=torch.zeros(4, 5),
        actions=torch.zeros(4, 2),
        rewards=torch.zeros(4, 1),
        costs=torch.zeros(4, 1),
        next_obs=torch.zeros(4, 5),
        dones=torch.zeros(4, 1),
    )

    diagnostics = agent.update(batch, environment_step=1, exploratory=False)

    assert diagnostics["cost_critic_loss"] == pytest.approx((0.99 * 0.8) ** 2)


class _DictObservationEnv(Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
                "dyn": spaces.Box(-np.inf, np.inf, shape=(42,), dtype=np.float32),
                "dyn_mask": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.counter = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "state": np.full(13, self.counter, dtype=np.float32),
            "dyn": np.arange(42, dtype=np.float32),
            "dyn_mask": np.ones(6, dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.counter = 0
        return self._obs(), {}

    def step(self, action):
        self.counter += 1
        return self._obs(), 1.0, False, False, {"reason": ""}


def test_trust_free_wrapper_preserves_continuous_action_and_resume_state() -> None:
    wrapped = TrustFreeObservationWrapper(_DictObservationEnv())
    obs, _ = wrapped.reset(seed=3)
    action = np.array([0.5, -0.25], dtype=np.float32)

    next_obs, _, _, _, _ = wrapped.step(action)
    saved = wrapped.get_resume_state()
    wrapped.step(action)
    wrapped.set_resume_state(saved)

    assert wrapped.observation_space.shape == (55,)
    assert wrapped.action_space.shape == (2,)
    assert obs.shape == (55,)
    assert next_obs.shape == (55,)
    assert wrapped.env.counter == 1


def test_replay_round_trip_preserves_contents_and_sampling_rng() -> None:
    replay = LagUReplayBuffer(capacity=5, obs_dim=3, action_dim=2, seed=17)
    replay.add_batch(
        obs=np.arange(9, dtype=np.float32).reshape(3, 3),
        actions=np.arange(6, dtype=np.float32).reshape(3, 2),
        rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        costs=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        next_obs=np.arange(9, 18, dtype=np.float32).reshape(3, 3),
        dones=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    restored = LagUReplayBuffer(capacity=5, obs_dim=3, action_dim=2, seed=999)
    restored.load_state_dict(replay.state_dict())

    first = replay.sample(4, device=torch.device("cpu"))
    second = restored.sample(4, device=torch.device("cpu"))

    assert restored.size == 3
    for field in first.__dict__:
        torch.testing.assert_close(getattr(first, field), getattr(second, field))


def test_agent_state_round_trip_restores_targets_optimizers_and_counters() -> None:
    torch.manual_seed(29)
    agent = LagUAgent(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(8, 8),
        policy_delay=1,
        target_policy_noise=0.0,
        target_noise_clip=0.0,
        device="cpu",
    )
    batch = ReplayBatch(
        obs=torch.randn(4, 5),
        actions=torch.empty(4, 2).uniform_(-1.0, 1.0),
        rewards=torch.randn(4, 1),
        costs=torch.ones(4, 1),
        next_obs=torch.randn(4, 5),
        dones=torch.zeros(4, 1),
    )
    agent.update(batch, environment_step=1, exploratory=False)
    probe = np.ones((2, 5), dtype=np.float32)
    expected = agent.act(probe)

    restored = LagUAgent(
        obs_dim=5,
        action_dim=2,
        hidden_dims=(8, 8),
        policy_delay=1,
        target_policy_noise=0.0,
        target_noise_clip=0.0,
        device="cpu",
    )
    restored.load_state_dict(agent.state_dict())

    np.testing.assert_allclose(restored.act(probe), expected)
    assert restored.gradient_steps == agent.gradient_steps
    assert restored.lagrange.item() == pytest.approx(agent.lagrange.item())
    assert restored.actor_optimizer.state_dict()["state"]


def test_lag_u_environment_config_disables_project_safety_modules() -> None:
    config = {
        "env": {"use_filter": False},
        "civo": {"enabled": True, "shield_enabled": True},
        "rc_colregs": {"enabled": True, "reward_weight": 4.0},
    }

    resolved = env_config_for_lag_u(config, "cache/path")

    assert resolved["env"]["use_filter"] is True
    assert resolved["cache"] == {"kf_cache_dir": "cache/path", "kf_cache_mode": "read_strict"}
    assert resolved["civo"] == {"enabled": False, "shield_enabled": False}
    assert resolved["rc_colregs"] == {"enabled": False, "reward_weight": 0.0}


def test_lag_u_timing_config_recomputes_filter_online() -> None:
    resolved = env_config_for_lag_u(
        {"env": {"use_filter": True}},
        "cache/path",
        online_exact=True,
    )

    assert resolved["env"]["filter_execution_mode"] == "online_exact"
    assert resolved["cache"]["kf_cache_mode"] == "read_strict"


def test_training_config_requires_equal_source_vecenv_and_aligned_steps() -> None:
    config = {
        "lag_u": {
            "n_envs": 4,
            "max_timesteps": 100,
            "learning_starts": 8,
            "batch_size": 8,
            "gamma": 0.99,
            "cost_gamma": 0.99,
        }
    }
    validate_training_config(config, source_count=4)

    with pytest.raises(ValueError, match="source count"):
        validate_training_config(config, source_count=3)
    with pytest.raises(ValueError, match="divisible"):
        validate_training_config(
            {
                "lag_u": {
                    "n_envs": 4,
                    "max_timesteps": 101,
                    "learning_starts": 8,
                    "batch_size": 8,
                    "cost_gamma": 0.99,
                }
            },
            source_count=4,
        )
    with pytest.raises(ValueError, match="cost_gamma"):
        validate_training_config(
            {
                "lag_u": {
                    "n_envs": 4,
                    "max_timesteps": 100,
                    "learning_starts": 8,
                    "batch_size": 8,
                    "gamma": 0.99,
                    "cost_gamma": 1.0,
                }
            },
            source_count=4,
        )


def test_eval_action_reports_reward_uncertainty_cost_and_adaptive_bound() -> None:
    actor = Actor(obs_dim=3, action_dim=2, hidden_dims=(4,))
    reward = RewardCriticEnsemble(obs_dim=3, action_dim=2, members=3, hidden_dims=(4,))
    cost = CostCritic(obs_dim=3, action_dim=2, hidden_dims=(4,))
    for parameter in actor.parameters():
        parameter.data.zero_()
    for index, member in enumerate(reward.members):
        for parameter in member.parameters():
            parameter.data.zero_()
        member.q1[-1].bias.data.fill_(float(index - 3))
    for parameter in cost.parameters():
        parameter.data.zero_()
    cost.network[-1].bias.data.fill_(0.25)

    action, diagnostics = select_action(
        actor,
        reward,
        cost,
        np.zeros(3, dtype=np.float32),
        device=torch.device("cpu"),
        base_cost_bound=0.1,
        uncertainty_threshold=0.07,
    )

    expected_std = float(np.sqrt(2.0 / 3.0))
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))
    assert diagnostics["q_mean"] == pytest.approx(-2.0)
    assert diagnostics["q_std"] == pytest.approx(expected_std)
    assert diagnostics["uncertainty_ratio"] == pytest.approx(expected_std / 2.0)
    assert diagnostics["cost_q"] == pytest.approx(0.25)
    assert diagnostics["adaptive_cost_bound"] == pytest.approx(0.1 * 0.07 / (expected_std / 2.0))


def test_eval_rejects_legacy_sigmoid_cost_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_version": 1,
            "config": {"lag_u": {"cost_gamma": 1.0}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="linear-cost"):
        load_model(checkpoint, torch.device("cpu"))
