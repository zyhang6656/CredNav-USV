from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F


COLLISION_REASONS = frozenset({"dynamic_obs", "static_obs", "out_of_bounds", "collision"})


def flatten_trust_free_observation(obs: dict[str, Any]) -> np.ndarray:
    state = np.asarray(obs["state"], dtype=np.float32)
    dyn = np.asarray(obs["dyn"], dtype=np.float32)
    mask = np.asarray(obs["dyn_mask"], dtype=np.float32)
    if state.shape[-1] != 13 or dyn.shape[-1] != 42 or mask.shape[-1] != 6:
        raise ValueError(
            f"expected state/dyn/mask dimensions 13/42/6, got "
            f"{state.shape[-1]}/{dyn.shape[-1]}/{mask.shape[-1]}"
        )
    prefix = state.shape[:-1]
    if dyn.shape[:-1] != prefix or mask.shape[:-1] != prefix:
        raise ValueError("state, dyn, and dyn_mask batch dimensions must match")
    dyn_without_trust = dyn.reshape(*prefix, 6, 7)[..., :6].reshape(*prefix, 36)
    return np.concatenate((state, dyn_without_trust, mask), axis=-1).astype(np.float32, copy=False)


def exploration_probability(step: int, *, tc: int = 300_000, minimum: float = 0.1) -> float:
    if tc <= 0:
        raise ValueError("tc must be positive")
    return float(max(float(minimum), 1.0 - float(step) / float(tc)))


def ensemble_statistics(member_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if member_values.ndim < 1 or member_values.shape[0] < 1:
        raise ValueError("member_values must have a non-empty ensemble dimension")
    mean = member_values.mean(dim=0)
    variance = member_values.square().mean(dim=0) - mean.square()
    return mean, variance.clamp_min(0.0)


def adaptive_cost_bound(
    uncertainty_ratio: torch.Tensor,
    *,
    base_bound: float = 0.1,
    threshold: float = 0.07,
) -> torch.Tensor:
    if base_bound < 0.0 or threshold <= 0.0:
        raise ValueError("base_bound must be nonnegative and threshold must be positive")
    base = torch.full_like(uncertainty_ratio, float(base_bound))
    reduced = float(base_bound) * float(threshold) / uncertainty_ratio.clamp_min(torch.finfo(uncertainty_ratio.dtype).eps)
    return torch.where(uncertainty_ratio <= float(threshold), base, reduced)


def transition_cost(info: dict[str, Any]) -> float:
    return float(str(info.get("reason", "")) in COLLISION_REASONS)


class TrustFreeObservationWrapper(gym.Wrapper):
    """Remove per-obstacle trust while preserving the native continuous action."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(55,),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        return flatten_trust_free_observation(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return flatten_trust_free_observation(obs), reward, terminated, truncated, info

    def get_resume_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.env.__dict__)

    def set_resume_state(self, state: dict[str, Any]) -> None:
        self.env.__dict__.update(copy.deepcopy(state))


def _glorot_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(input_dim)
    for hidden in hidden_dims:
        layers.extend((nn.Linear(last, int(hidden)), nn.ReLU()))
        last = int(hidden)
    layers.append(nn.Linear(last, int(output_dim)))
    network = nn.Sequential(*layers)
    network.apply(_glorot_init)
    return network


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.network = _mlp(obs_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs.float()))


class TwinRewardCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        input_dim = int(obs_dim) + int(action_dim)
        self.q1 = _mlp(input_dim, hidden_dims, 1)
        self.q2 = _mlp(input_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((obs.float(), action.float()), dim=-1)
        return self.q1(features), self.q2(features)


class RewardCriticEnsemble(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        members: int = 3,
        hidden_dims: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        if members <= 0:
            raise ValueError("members must be positive")
        self.members = nn.ModuleList(
            TwinRewardCritic(obs_dim, action_dim, hidden_dims) for _ in range(int(members))
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [member(obs, action) for member in self.members]
        return torch.stack([item[0] for item in outputs]), torch.stack([item[1] for item in outputs])

    def policy_values(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, _ = self(obs, action)
        return q1


class CostCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.network = _mlp(int(obs_dim) + int(action_dim), hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((obs.float(), action.float()), dim=-1))


def magnitude_epsilon_denominator(value: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(value.dtype).eps
    return value.abs().clamp_min(epsilon)


def stable_standard_deviation(variance: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(variance.dtype).eps
    root = variance.clamp_min(epsilon).sqrt()
    return torch.where(variance > epsilon, root, torch.zeros_like(root))


def uncertainty_ratio(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    return stable_standard_deviation(variance) / magnitude_epsilon_denominator(mean)


def discounted_target(
    immediate: torch.Tensor,
    next_value: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    return immediate + float(gamma) * (1.0 - done) * next_value


def actor_objective(
    reward_member_values: torch.Tensor,
    cost_values: torch.Tensor,
    lagrange: torch.Tensor,
    *,
    exploratory: bool,
    base_bound: float = 0.1,
    uncertainty_threshold: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reward_mean, reward_variance = ensemble_statistics(reward_member_values)
    reward_std = stable_standard_deviation(reward_variance)
    ratio = reward_std / magnitude_epsilon_denominator(reward_mean)
    if exploratory:
        bound = torch.full_like(cost_values, float(base_bound))
        loss = -reward_mean - reward_std + lagrange * (cost_values - bound)
    else:
        bound = adaptive_cost_bound(
            ratio,
            base_bound=base_bound,
            threshold=uncertainty_threshold,
        )
        loss = -reward_mean + lagrange * (cost_values - bound)
    return loss.mean(), bound, ratio


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor

    def to(self, device: torch.device) -> "ReplayBatch":
        return ReplayBatch(*(value.to(device) for value in self.__dict__.values()))


class LagUReplayBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, action_dim: int, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.costs = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(int(seed))

    def add_batch(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim)
        batch_size = int(obs.shape[0])
        actions = np.asarray(actions, dtype=np.float32).reshape(batch_size, self.action_dim)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(batch_size, 1)
        costs = np.asarray(costs, dtype=np.float32).reshape(batch_size, 1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(batch_size, self.obs_dim)
        dones = np.asarray(dones, dtype=np.float32).reshape(batch_size, 1)
        if batch_size > self.capacity:
            start = batch_size - self.capacity
            obs, actions, rewards = obs[start:], actions[start:], rewards[start:]
            costs, next_obs, dones = costs[start:], next_obs[start:], dones[start:]
            batch_size = self.capacity
        indices = (self.position + np.arange(batch_size)) % self.capacity
        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.costs[indices] = costs
        self.next_obs[indices] = next_obs
        self.dones[indices] = dones
        self.position = int((self.position + batch_size) % self.capacity)
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int, *, device: torch.device) -> ReplayBatch:
        if self.size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        values = (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.costs[indices],
            self.next_obs[indices],
            self.dones[indices],
        )
        return ReplayBatch(
            *(torch.as_tensor(value, dtype=torch.float32, device=device) for value in values)
        )

    def state_dict(self) -> dict[str, Any]:
        stored = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "obs": self.obs[:stored],
            "actions": self.actions[:stored],
            "rewards": self.rewards[:stored],
            "costs": self.costs[:stored],
            "next_obs": self.next_obs[:stored],
            "dones": self.dones[:stored],
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.capacity, self.obs_dim, self.action_dim)
        actual = (int(state["capacity"]), int(state["obs_dim"]), int(state["action_dim"]))
        if actual != expected:
            raise ValueError(f"replay dimensions {actual} do not match {expected}")
        for name in ("obs", "actions", "rewards", "costs", "next_obs", "dones"):
            destination = getattr(self, name)
            source = np.asarray(state[name], dtype=np.float32)
            expected_shape = (int(state["size"]), *destination.shape[1:])
            if int(state["size"]) == self.capacity:
                expected_shape = destination.shape
            if source.shape != expected_shape:
                raise ValueError(f"replay {name} shape {source.shape} does not match {expected_shape}")
            np.copyto(destination[: source.shape[0]], source)
        self.position = int(state["position"])
        self.size = int(state["size"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class LagUAgent:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        ensemble_members: int = 3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        lagrange_lr: float = 1e-5,
        gamma: float = 0.99,
        cost_gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        base_cost_bound: float = 0.1,
        uncertainty_threshold: float = 0.07,
        exploration_tc: int = 300_000,
        exploration_minimum: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.reward_critic = RewardCriticEnsemble(
            obs_dim,
            action_dim,
            ensemble_members,
            hidden_dims,
        ).to(self.device)
        self.cost_critic = CostCritic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device).requires_grad_(False)
        self.target_reward_critic = copy.deepcopy(self.reward_critic).to(self.device).requires_grad_(False)
        self.target_cost_critic = copy.deepcopy(self.cost_critic).to(self.device).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.reward_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=float(critic_lr))
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=float(critic_lr))
        self.lagrange = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        self.lagrange_optimizer = torch.optim.Adam([self.lagrange], lr=float(lagrange_lr))
        self.gamma = float(gamma)
        self.cost_gamma = float(cost_gamma)
        self.tau = float(tau)
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.base_cost_bound = float(base_cost_bound)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.exploration_tc = int(exploration_tc)
        self.exploration_minimum = float(exploration_minimum)
        self.action_dim = int(action_dim)
        self.gradient_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor(obs_tensor).cpu().numpy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_reward_critic": self.target_reward_critic.state_dict(),
            "target_cost_critic": self.target_cost_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "reward_optimizer": self.reward_optimizer.state_dict(),
            "cost_optimizer": self.cost_optimizer.state_dict(),
            "lagrange_optimizer": self.lagrange_optimizer.state_dict(),
            "lagrange": float(self.lagrange.detach().item()),
            "gradient_steps": self.gradient_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.reward_critic.load_state_dict(state["reward_critic"])
        self.cost_critic.load_state_dict(state["cost_critic"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_reward_critic.load_state_dict(state["target_reward_critic"])
        self.target_cost_critic.load_state_dict(state["target_cost_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.reward_optimizer.load_state_dict(state["reward_optimizer"])
        self.cost_optimizer.load_state_dict(state["cost_optimizer"])
        self.lagrange_optimizer.load_state_dict(state["lagrange_optimizer"])
        with torch.no_grad():
            self.lagrange.fill_(float(state["lagrange"]))
        self.gradient_steps = int(state["gradient_steps"])
        for optimizer in (
            self.actor_optimizer,
            self.reward_optimizer,
            self.cost_optimizer,
            self.lagrange_optimizer,
        ):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(self.device)

    def update(
        self,
        batch: ReplayBatch,
        *,
        environment_step: int,
        exploratory: bool | None = None,
    ) -> dict[str, float | bool]:
        batch = batch.to(self.device)
        with torch.no_grad():
            clean_next_actions = self.target_actor(batch.next_obs)
            reward_next_actions = clean_next_actions
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(reward_next_actions) * self.target_policy_noise
                noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
                reward_next_actions = (reward_next_actions + noise).clamp(-1.0, 1.0)
            next_q1, next_q2 = self.target_reward_critic(batch.next_obs, reward_next_actions)
            reward_targets = discounted_target(
                batch.rewards.unsqueeze(0),
                torch.minimum(next_q1, next_q2),
                batch.dones.unsqueeze(0),
                gamma=self.gamma,
            )
            cost_targets = discounted_target(
                batch.costs,
                self.target_cost_critic(batch.next_obs, clean_next_actions),
                batch.dones,
                gamma=self.cost_gamma,
            )

        current_q1, current_q2 = self.reward_critic(batch.obs, batch.actions)
        reward_loss = sum(
            F.mse_loss(current_q1[index], reward_targets[index])
            + F.mse_loss(current_q2[index], reward_targets[index])
            for index in range(len(self.reward_critic.members))
        )
        self.reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        self.reward_optimizer.step()

        current_cost = self.cost_critic(batch.obs, batch.actions)
        cost_loss = F.mse_loss(current_cost, cost_targets)
        self.cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        self.cost_optimizer.step()

        self.gradient_steps += 1
        diagnostics: dict[str, float | bool] = {
            "reward_critic_loss": float(reward_loss.detach().item()),
            "cost_critic_loss": float(cost_loss.detach().item()),
            "actor_updated": False,
        }
        if self.gradient_steps % self.policy_delay != 0:
            return diagnostics

        if exploratory is None:
            probability = exploration_probability(
                environment_step,
                tc=self.exploration_tc,
                minimum=self.exploration_minimum,
            )
            exploratory = random.random() < probability

        self.reward_critic.requires_grad_(False)
        self.cost_critic.requires_grad_(False)
        policy_actions = self.actor(batch.obs)
        member_values = self.reward_critic.policy_values(batch.obs, policy_actions)
        policy_cost = self.cost_critic(batch.obs, policy_actions)
        actor_loss, cost_bound, ratio = actor_objective(
            member_values,
            policy_cost,
            self.lagrange.detach(),
            exploratory=bool(exploratory),
            base_bound=self.base_cost_bound,
            uncertainty_threshold=self.uncertainty_threshold,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.reward_critic.requires_grad_(True)
        self.cost_critic.requires_grad_(True)

        lambda_loss = -(self.lagrange * (policy_cost.detach() - cost_bound.detach())).mean()
        self.lagrange_optimizer.zero_grad(set_to_none=True)
        lambda_loss.backward()
        self.lagrange_optimizer.step()
        with torch.no_grad():
            self.lagrange.clamp_(min=0.0)

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.reward_critic, self.target_reward_critic)
        self._soft_update(self.cost_critic, self.target_cost_critic)
        diagnostic_mean, diagnostic_variance = ensemble_statistics(member_values.detach())
        diagnostic_std = stable_standard_deviation(diagnostic_variance)
        diagnostics.update(
            {
                "actor_updated": True,
                "exploratory_update": bool(exploratory),
                "actor_loss": float(actor_loss.detach().item()),
                "lagrange": float(self.lagrange.detach().item()),
                "q_std_mean": float(diagnostic_std.mean().item()),
                "uncertainty_ratio_mean": float(ratio.detach().mean().item()),
                "uncertainty_ratio_p95": float(torch.quantile(ratio.detach().reshape(-1), 0.95).item()),
                "cost_q_mean": float(policy_cost.detach().mean().item()),
                "cost_bound_mean": float(cost_bound.detach().mean().item()),
                "nonpositive_q_mean_rate": float((diagnostic_mean <= 0.0).float().mean().item()),
            }
        )
        return diagnostics

    @torch.no_grad()
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)
