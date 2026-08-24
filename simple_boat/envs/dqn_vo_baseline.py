from __future__ import annotations

import copy
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def tcpa_dcpa(
    dx: float,
    dy: float,
    own_vx: float,
    own_vy: float,
    obs_vx: float,
    obs_vy: float,
) -> tuple[float, float]:
    rvx = float(obs_vx) - float(own_vx)
    rvy = float(obs_vy) - float(own_vy)
    v2 = rvx * rvx + rvy * rvy
    if v2 < 1e-9:
        return -1.0, float(math.hypot(dx, dy))
    tcpa = -((float(dx) * rvx + float(dy) * rvy) / v2)
    if tcpa <= 0.0:
        return float(tcpa), float(math.hypot(dx, dy))
    return float(tcpa), float(math.hypot(float(dx) + rvx * tcpa, float(dy) + rvy * tcpa))


def risk_score(tcpa: float, dcpa: float, horizon: float, warning_distance: float) -> float:
    if tcpa <= 0.0 or tcpa > horizon or dcpa >= warning_distance:
        return 0.0
    spatial = 1.0 - float(dcpa) / max(float(warning_distance), 1e-9)
    temporal = 1.0 - float(tcpa) / max(float(horizon), 1e-9)
    return float(np.clip(0.5 * spatial + 0.5 * temporal, 0.0, 1.0))


def velocity_inside_vo(
    dx: float,
    dy: float,
    own_vx: float,
    own_vy: float,
    obs_vx: float,
    obs_vy: float,
    radius: float,
) -> bool:
    dist = math.hypot(float(dx), float(dy))
    radius = max(float(radius), 1e-9)
    if dist <= radius:
        return True
    rel_vx = float(own_vx) - float(obs_vx)
    rel_vy = float(own_vy) - float(obs_vy)
    rel_speed = math.hypot(rel_vx, rel_vy)
    if rel_speed < 1e-9:
        return False
    closing = float(dx) * rel_vx + float(dy) * rel_vy
    if closing <= 0.0:
        return False
    cosang = closing / max(dist * rel_speed, 1e-9)
    half_angle = math.asin(min(1.0, radius / dist))
    return bool(cosang >= math.cos(half_angle))


def starboard_required(dx: float, dy: float) -> bool:
    beta = math.atan2(float(dy), float(dx))
    return bool(abs(beta) <= math.radians(22.5) or (dx > 0.0 and dy < 0.0))


class DQNVOObservationRewardWrapper(gym.Wrapper):
    """Adapt USVEnvMinimal into the DQN-VO baseline from Li et al."""

    def __init__(
        self,
        env: gym.Env,
        fixed_surge: float = 0.7,
        yaw_command: float = 0.45,
        tcpa_horizon: float = 12.0,
        warning_distance: float = 4.0,
    ):
        super().__init__(env)
        self.fixed_surge = float(fixed_surge)
        self.yaw_command = float(yaw_command)
        self.tcpa_horizon = float(tcpa_horizon)
        self.warning_distance = float(warning_distance)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(24,), dtype=np.float32)
        self._prev_goal_distance = 0.0

    def map_action(self, action: int | np.ndarray) -> np.ndarray:
        idx = int(np.asarray(action).reshape(-1)[0])
        yaw = {0: self.yaw_command, 1: 0.0, 2: -self.yaw_command}[idx]
        return np.array([self.fixed_surge, yaw], dtype=np.float32)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_distance = self._goal_distance()
        return self._flat_obs(obs), info

    def step(self, action: int | np.ndarray):
        idx = int(np.asarray(action).reshape(-1)[0])
        obs, _base_reward, terminated, truncated, info = self.env.step(self.map_action(idx))
        reward, parts = self._dqn_vo_reward(info, idx)
        info.update(parts)
        return self._flat_obs(obs), reward, terminated, truncated, info

    def get_resume_state(self) -> dict[str, Any]:
        return {
            "env": copy.deepcopy(self.env.__dict__),
            "prev_goal_distance": float(self._prev_goal_distance),
        }

    def set_resume_state(self, state: dict[str, Any]) -> None:
        self.env.__dict__.update(copy.deepcopy(state["env"]))
        self._prev_goal_distance = float(state["prev_goal_distance"])

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(np.asarray(self.env.goal, dtype=float) - self.env.ship_state[:2]))

    def _select_target(self) -> dict[str, float]:
        own_u = float(self.env.ship_state[3])
        own_v = float(self.env.ship_state[4])
        best: dict[str, float] | None = None
        for est in getattr(self.env, "obstacle_estimates", {}).values():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            vx = float(est.get("vx", 0.0))
            vy = float(est.get("vy", 0.0))
            tcpa, dcpa = tcpa_dcpa(dx, dy, own_u, own_v, vx, vy)
            risk = risk_score(tcpa, dcpa, self.tcpa_horizon, self.warning_distance)
            item = {"dx": dx, "dy": dy, "vx": vx, "vy": vy, "tcpa": tcpa, "dcpa": dcpa, "risk": risk}
            if best is None or item["risk"] > best["risk"]:
                best = item
        return best or {"dx": -1.0, "dy": 0.0, "vx": 0.0, "vy": 0.0, "tcpa": -1.0, "dcpa": 10.0, "risk": 0.0}

    def _flat_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[:13]
        target = self._select_target()
        pos_scale = max(float(getattr(self.env, "obs_norm_range", 10.0)), 1e-6)
        v_scale = max(float(getattr(self.env, "u_max", 3.0)), 1e-6)
        dyn = np.array(
            [
                np.clip(target["dx"] / pos_scale, -1.0, 1.0),
                np.clip(target["dy"] / pos_scale, -1.0, 1.0),
                np.clip(target["vx"] / v_scale, -1.0, 1.0),
                np.clip(target["vy"] / v_scale, -1.0, 1.0),
                np.clip(math.hypot(target["dx"], target["dy"]) / pos_scale, 0.0, 1.0),
                np.clip(target["tcpa"] / max(self.tcpa_horizon, 1e-6), -1.0, 1.0),
                np.clip(target["dcpa"] / max(self.warning_distance, 1e-6), 0.0, 1.0),
                np.clip(target["risk"], 0.0, 1.0),
                1.0 if starboard_required(target["dx"], target["dy"]) else 0.0,
                1.0 if target["risk"] > 0.0 else 0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([state, dyn]).astype(np.float32)

    def _dqn_vo_reward(self, info: dict[str, Any], action: int) -> tuple[float, dict[str, float]]:
        cur_goal = float(info.get("distance_to_goal", self._goal_distance()))
        progress = self._prev_goal_distance - cur_goal
        self._prev_goal_distance = cur_goal

        target = self._select_target()
        own_u = float(self.env.ship_state[3])
        own_v = float(self.env.ship_state[4])
        radius = float(getattr(self.env, "R_usv", 1.0) + getattr(self.env, "dyn_radius", 1.0))
        inside_vo = velocity_inside_vo(target["dx"], target["dy"], own_u, own_v, target["vx"], target["vy"], radius)
        yaw_action = float(self.map_action(action)[1])
        colregs_violation = 1.0 if starboard_required(target["dx"], target["dy"]) and yaw_action > -1e-6 else 0.0
        reason = str(info.get("reason", ""))
        collision = 1.0 if reason in {"dynamic_obs", "static_obs", "out_of_bounds", "collision"} else 0.0
        success = 1.0 if reason == "goal_reached" else 0.0

        reward = (
            30.0 * progress
            + 200.0 * success
            - 300.0 * collision
            - 3.0 * float(target["risk"])
            - 1.8 * colregs_violation
            - 1.0 * float(inside_vo)
            - 0.01
        )
        return float(reward), {
            "dqnvo_risk": float(target["risk"]),
            "dqnvo_inside_vo": float(inside_vo),
            "dqnvo_colregs_violation": float(colregs_violation),
            "dqnvo_reward": float(reward),
        }
