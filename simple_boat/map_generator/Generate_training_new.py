"""Long-interaction scenario generator for CW-VL experiments.

This generator keeps the original V2 idea of simple constant-velocity
dynamic obstacles that stop at the map boundary, but adds explicit timing
constraints. The goal is to create dynamic interactions around the learned
obs3 episode length rather than only early encounters.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

H = W = 32
DT = 0.1
T = 640
T_CHECK = 200
DOMAIN_MIN = 1.0
DOMAIN_MAX_X = W - 1.0
DOMAIN_MAX_Y = H - 1.0
LEARNED_ARRIVAL_STEPS = 230


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _inside_domain(x: float, y: float) -> bool:
    return (DOMAIN_MIN <= x <= DOMAIN_MAX_X) and (DOMAIN_MIN <= y <= DOMAIN_MAX_Y)


def _unit_from_angle(psi: float) -> np.ndarray:
    return np.array([math.cos(psi), math.sin(psi)], dtype=np.float32)


def _distance_to_boundary(pos: np.ndarray, direction: np.ndarray) -> float:
    """Distance from pos to the first map boundary hit along direction."""
    x, y = float(pos[0]), float(pos[1])
    dx, dy = float(direction[0]), float(direction[1])
    candidates = []
    if dx > 1e-8:
        candidates.append((DOMAIN_MAX_X - x) / dx)
    elif dx < -1e-8:
        candidates.append((DOMAIN_MIN - x) / dx)
    if dy > 1e-8:
        candidates.append((DOMAIN_MAX_Y - y) / dy)
    elif dy < -1e-8:
        candidates.append((DOMAIN_MIN - y) / dy)
    candidates = [c for c in candidates if c > 0.0]
    return min(candidates) if candidates else float("inf")


class TrainingSceneGenerator:
    """Shared base logic used by the long-interaction generator.

    The original V2 generator remains as a historical standalone entry point,
    but this file is self-contained so the current long-interaction map path
    can be kept, tested, or archived independently.
    """

    def __init__(self, n_obs: Optional[int] = None, colregs_mode: bool = False):
        self.usv_speed_nom = 2.0
        self.usv_start_x = 5.0
        self.usv_start_y_range = (4.0, 12.0)
        self.usv_heading_noise = math.radians(5)
        self.usv_goal = np.array([27.0, 27.0], dtype=np.float32)

        if n_obs is not None:
            self.n_obs = max(0, int(n_obs))
            self.random_count = False
        else:
            self.n_obs = None
            self.random_count = True
        self.num_obs_choices = [1, 2, 3, 4, 5, 6]
        self.num_obs_weights = [0.10, 0.18, 0.24, 0.22, 0.16, 0.10]

        _n = max(n_obs, 1) if n_obs is not None else 3
        self.min_obstacle_distance = max(2.0, 4.0 - 0.4 * _n)
        self.min_obstacle_to_start = max(4.0, 8.0 - 0.6 * _n)
        self.max_tries = 350
        self.min_obs_obs_dist = 1.2
        self.colregs_mode = colregs_mode

    @staticmethod
    def _traj_positions(start_pos: np.ndarray, traj: np.ndarray) -> np.ndarray:
        n = len(traj)
        pos = np.zeros((n + 1, 2), dtype=np.float32)
        pos[0] = start_pos[:2]
        pos[1:] = traj[:, 0:2]
        return pos

    @staticmethod
    def _min_distance_two_piecewise_linear(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        min_d2 = float("inf")
        n = min(len(pos_a), len(pos_b), T_CHECK + 1) - 1
        for k in range(n):
            a0, a1 = pos_a[k], pos_a[k + 1]
            b0, b1 = pos_b[k], pos_b[k + 1]
            va = (a1 - a0) / DT
            vb = (b1 - b0) / DT
            p0 = a0 - b0
            v = va - vb
            vv = float(v[0] * v[0] + v[1] * v[1])
            if vv < 1e-12:
                tau = 0.0
            else:
                pv = float(p0[0] * v[0] + p0[1] * v[1])
                tau = -pv / vv
                tau = 0.0 if tau < 0.0 else (DT if tau > DT else tau)
            d0 = p0 + v * tau
            d2 = float(d0[0] * d0[0] + d0[1] * d0[1])
            if d2 < min_d2:
                min_d2 = d2
        return float(math.sqrt(min_d2))

    def _collides_with_existing(
        self,
        cand_start: np.ndarray,
        cand_traj: np.ndarray,
        existing: List[Dict],
        existing_trajs: List[np.ndarray],
    ) -> bool:
        if len(existing) == 0:
            return False
        pos_c = self._traj_positions(cand_start, cand_traj)
        thr = float(self.min_obs_obs_dist)
        for e, traj_e in zip(existing, existing_trajs):
            pos_e = self._traj_positions(e["start_pos"], traj_e)
            if self._min_distance_two_piecewise_linear(pos_c, pos_e) < thr:
                return True
        return False

    def _generate_linear_obstacle_trajectory(
        self, start_pos: np.ndarray, velocity: float, heading: float
    ) -> np.ndarray:
        traj = np.zeros((T, 4), dtype=np.float32)
        x, y = float(start_pos[0]), float(start_pos[1])
        psi, v = float(heading), float(velocity)
        inside = _inside_domain(x, y)
        stopped = False
        stop_x = stop_y = 0.0
        eps = 1e-5
        for t in range(T):
            if stopped:
                traj[t] = [stop_x, stop_y, eps * math.cos(psi), eps * math.sin(psi)]
                continue
            vx = v * math.cos(psi)
            vy = v * math.sin(psi)
            x_new = x + vx * DT
            y_new = y + vy * DT
            now_inside = _inside_domain(x_new, y_new)
            if inside and not now_inside:
                stopped = True
                stop_x, stop_y = x, y
                traj[t] = [stop_x, stop_y, eps * math.cos(psi), eps * math.sin(psi)]
            else:
                x, y = x_new, y_new
                inside = now_inside
                traj[t] = [x, y, vx, vy]
        return traj

    def _generate_usv_nominal_trajectory(self, start_pos: np.ndarray, heading: float) -> np.ndarray:
        traj = np.zeros((T, 4), dtype=np.float32)
        x, y = float(start_pos[0]), float(start_pos[1])
        psi, v = float(heading), float(self.usv_speed_nom)
        for t in range(T):
            traj[t] = [x, y, v * math.cos(psi), v * math.sin(psi)]
            x = _clamp(x + v * math.cos(psi) * DT, DOMAIN_MIN, DOMAIN_MAX_X)
            y = _clamp(y + v * math.sin(psi) * DT, DOMAIN_MIN, DOMAIN_MAX_Y)
        return traj

    def generate_scenes(self, num_scenes: int = 500) -> List[Dict]:
        scenes: List[Dict] = []
        for i in range(num_scenes):
            if (i + 1) % 50 == 0:
                print(f"--- Generating scene {i + 1}/{num_scenes} ---", flush=True)
            scene = None
            retry = 0
            while scene is None:
                scene = self._generate_one_scene()
                retry += 1
                if retry % 500 == 0:
                    print(f"[WARN] scene {i}: retry={retry}", flush=True)
            scenes.append(scene)
        return scenes


class LongInteractionSceneGenerator(TrainingSceneGenerator):
    """Timing-constrained version of the original training generator."""

    def __init__(self, n_obs: Optional[int] = None, colregs_mode: bool = False):
        super().__init__(n_obs=n_obs, colregs_mode=colregs_mode)
        self.headon_speed_range = (0.8, 2.5)
        self.slow_wall_speed_range = (0.35, 2.5)
        self.sm_speed_range = (0.6, 2.2)
        self.max_timing_tries = 900

    def _learned_proxy_speed(self, usv_start: np.ndarray, usv_goal: np.ndarray) -> float:
        dist = float(np.linalg.norm(usv_goal - usv_start))
        return dist / (LEARNED_ARRIVAL_STEPS * DT)

    def _plan_timing_roles(self, n_obs: int) -> List[str]:
        if n_obs == 4:
            roles = ["early_threat", "mid_threat", "mid_threat", "late_threat"]
            random.shuffle(roles)
            return roles

        if n_obs == 5:
            roles = [
                "early_threat",
                "mid_threat",
                "mid_threat",
                "late_threat",
                "persistent_wall",
            ]
            random.shuffle(roles)
            return roles

        if n_obs == 6:
            roles = [
                "early_threat",
                "mid_threat",
                "mid_threat",
                "mid_threat",
                "late_threat",
                "persistent_wall",
            ]
            random.shuffle(roles)
            return roles

        if n_obs <= 3:
            late_count = n_obs if n_obs <= 2 else 2
            roles = ["late_threat"] * late_count
            roles.extend(random.choices(
                ["mid_threat", "persistent_wall"],
                weights=[0.65, 0.35],
                k=n_obs - late_count,
            ))
            random.shuffle(roles)
            return roles

        early_count = random.randint(1, min(2, n_obs - 1))
        roles = ["early_threat"] * early_count
        remaining = n_obs - early_count
        roles.extend(random.choices(
            ["mid_threat", "late_threat", "persistent_wall"],
            weights=[0.35, 0.45, 0.20],
            k=remaining,
        ))
        if not any(r == "late_threat" for r in roles):
            non_early = [i for i, r in enumerate(roles) if r != "early_threat"]
            roles[random.choice(non_early)] = "late_threat"
        random.shuffle(roles)
        return roles

    @staticmethod
    def _role_bounds(role: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if role == "early_threat":
            return (60, 90), (120, 170)
        if role == "mid_threat":
            return (100, 170), (200, 240)
        if role == "late_threat":
            return (180, 220), (220, 240)
        if role == "persistent_wall":
            return (70, 220), (200, 240)
        raise ValueError(f"Unknown timing role: {role}")

    @staticmethod
    def _role_bounds_for_count(role: str, n_obs: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if n_obs == 4:
            if role == "early_threat":
                return (70, 90), (120, 170)
            if role == "mid_threat":
                return (120, 190), (220, 250)
            if role == "late_threat":
                return (200, 240), (240, 270)
        if n_obs == 5:
            if role == "early_threat":
                return (70, 90), (120, 170)
            if role == "mid_threat":
                return (120, 190), (220, 250)
            if role == "late_threat":
                return (200, 240), (240, 270)
            if role == "persistent_wall":
                return (100, 220), (220, 270)
        if n_obs == 6:
            if role == "early_threat":
                return (70, 90), (120, 170)
            if role == "mid_threat":
                return (110, 190), (220, 250)
            if role == "late_threat":
                return (200, 240), (240, 270)
            if role == "persistent_wall":
                return (80, 230), (220, 270)
        return LongInteractionSceneGenerator._role_bounds(role)

    def _choose_type_for_role(self, role: str) -> str:
        if role == "early_threat":
            return random.choice(["single_headon", "slow_wall"])
        if role == "mid_threat":
            return random.choices(
                ["slow_wall", "single_headon", "speed_matched"],
                weights=[0.45, 0.20, 0.35],
                k=1,
            )[0]
        if role == "late_threat":
            return random.choices(
                ["slow_wall", "speed_matched"],
                weights=[0.60, 0.40],
                k=1,
            )[0]
        return "slow_wall"

    def _heading_for_type(self, obs_type: str, usv_psi: float, usv_dir: np.ndarray) -> float:
        if obs_type == "single_headon":
            base = math.atan2(float(-usv_dir[1]), float(-usv_dir[0]))
            return float((base + random.uniform(-math.radians(12), math.radians(12))) % (2 * math.pi))
        if obs_type == "speed_matched":
            return float((usv_psi + random.uniform(-math.radians(12), math.radians(12))) % (2 * math.pi))
        side = random.choice([-1.0, 1.0])
        return float((usv_psi + side * random.uniform(math.radians(55), math.radians(115))) % (2 * math.pi))

    @staticmethod
    def _speed_range_for_type(obs_type: str) -> Tuple[float, float]:
        if obs_type == "single_headon":
            return 0.8, 2.5
        if obs_type == "speed_matched":
            return 0.6, 2.2
        return 0.35, 2.5

    def _sample_timed_obstacle(
        self,
        role: str,
        usv_start: np.ndarray,
        usv_psi: float,
        usv_dir: np.ndarray,
        usv_goal: np.ndarray,
        existing: List[Dict],
        obs_idx: int,
        n_obs: int,
    ) -> Optional[Dict]:
        conflict_bounds, active_bounds = self._role_bounds_for_count(role, n_obs)
        learned_speed = self._learned_proxy_speed(usv_start, usv_goal)
        n_left = np.array([-usv_dir[1], usv_dir[0]], dtype=np.float32)

        for _ in range(self.max_timing_tries):
            active_until = random.randint(*active_bounds)
            c_hi = min(conflict_bounds[1], active_until - 18)
            if c_hi < conflict_bounds[0]:
                continue
            conflict_step = random.randint(conflict_bounds[0], c_hi)
            obs_type = self._choose_type_for_role(role)

            path_center = usv_start + learned_speed * DT * conflict_step * usv_dir
            if not (2.0 <= float(path_center[0]) <= W - 2.0 and 2.0 <= float(path_center[1]) <= H - 2.0):
                continue

            if obs_type == "single_headon":
                lateral = random.uniform(-0.4, 0.4)
            elif obs_type == "speed_matched":
                lateral = random.uniform(-1.0, 1.0)
            else:
                lateral = random.uniform(-1.5, 1.5)
            conflict_pos = path_center + lateral * n_left
            if not _inside_domain(float(conflict_pos[0]), float(conflict_pos[1])):
                continue

            heading = self._heading_for_type(obs_type, usv_psi, usv_dir)
            direction = _unit_from_angle(heading)
            boundary_dist = _distance_to_boundary(conflict_pos, direction)
            if not np.isfinite(boundary_dist) or boundary_dist <= 0:
                continue

            remaining_time = (active_until - conflict_step) * DT
            if remaining_time <= 0:
                continue
            speed = boundary_dist / remaining_time
            speed_lo, speed_hi = self._speed_range_for_type(obs_type)
            if not (speed_lo <= speed <= speed_hi):
                continue

            start_pos = conflict_pos - speed * conflict_step * DT * direction
            if not _inside_domain(float(start_pos[0]), float(start_pos[1])):
                continue
            if float(np.linalg.norm(start_pos - usv_start)) < self.min_obstacle_to_start:
                continue
            if any(float(np.linalg.norm(start_pos - e["start_pos"])) < self.min_obstacle_distance
                   for e in existing):
                continue

            traj = self._generate_linear_obstacle_trajectory(start_pos.astype(np.float32), speed, heading)
            measured_active_until = self._measure_active_until(traj)
            if not (active_bounds[0] <= measured_active_until <= active_bounds[1]):
                continue
            conflict_error = float(np.linalg.norm(traj[conflict_step, :2] - conflict_pos))
            if conflict_error > 0.35:
                continue

            return {
                "start_pos": start_pos.astype(np.float32),
                "velocity": float(speed),
                "heading": float(heading),
                "meet_time": float(conflict_step * DT),
                "meet_position": conflict_pos.astype(np.float32),
                "type": obs_type,
                "timing_role": role,
                "conflict_step": int(conflict_step),
                "active_until_step": int(measured_active_until),
                "target_active_until_step": int(active_until),
                "obs_idx": int(obs_idx),
                "seed": random.randint(0, 2**31 - 1),
            }
        return None

    @staticmethod
    def _measure_active_until(traj: np.ndarray) -> int:
        speeds = np.linalg.norm(traj[:, 2:4], axis=1)
        active = speeds > 1e-4
        if not bool(active.any()):
            return 0
        return int(np.where(active)[0][-1] + 1)

    def _generate_one_scene(self) -> Optional[Dict]:
        start_y = random.uniform(*self.usv_start_y_range)
        usv_start = np.array([self.usv_start_x, start_y], dtype=np.float32)
        usv_goal = self.usv_goal.copy()
        ideal_psi = math.atan2(float(usv_goal[1] - usv_start[1]), float(usv_goal[0] - usv_start[0]))
        usv_psi = ideal_psi + random.uniform(-self.usv_heading_noise, self.usv_heading_noise)
        usv_dir = _unit_from_angle(usv_psi)
        n_obs = self.n_obs if not self.random_count else random.choices(
            self.num_obs_choices, weights=self.num_obs_weights, k=1
        )[0]

        grid = np.zeros((H, W), dtype=np.uint8)
        roles = self._plan_timing_roles(int(n_obs))
        obstacles: List[Dict] = []
        obs_trajs: List[np.ndarray] = []

        for obs_idx, role in enumerate(roles):
            success = False
            for _ in range(self.max_tries):
                cand = self._sample_timed_obstacle(
                    role, usv_start, usv_psi, usv_dir, usv_goal, obstacles, obs_idx, int(n_obs)
                )
                if cand is None:
                    continue
                traj = self._generate_linear_obstacle_trajectory(
                    cand["start_pos"], cand["velocity"], cand["heading"]
                )
                if self._collides_with_existing(cand["start_pos"], traj, obstacles, obs_trajs):
                    continue
                obstacles.append(cand)
                obs_trajs.append(traj)
                success = True
                break
            if not success:
                return None

        if len(obstacles) != int(n_obs):
            return None

        usv_traj_nom = self._generate_usv_nominal_trajectory(usv_start, usv_psi)
        return {
            "grid": grid,
            "usv_start": usv_start,
            "usv_goal": usv_goal,
            "usv_psi": float(usv_psi),
            "usv_trajectory": usv_traj_nom,
            "obstacles": obstacles,
            "obstacle_trajectories": obs_trajs,
        }

    @staticmethod
    def build_metadata(scene: Dict) -> Dict[str, np.ndarray]:
        obstacles = scene["obstacles"]
        return {
            "obs_types": np.array([o["type"] for o in obstacles], dtype="U32"),
            "timing_roles": np.array([o["timing_role"] for o in obstacles], dtype="U32"),
            "conflict_steps": np.array([o["conflict_step"] for o in obstacles], dtype=np.int32),
            "active_until_steps": np.array([o["active_until_step"] for o in obstacles], dtype=np.int32),
            "target_active_until_steps": np.array(
                [o["target_active_until_step"] for o in obstacles], dtype=np.int32
            ),
        }


def save_scenes_new(scenes: List[Dict], base_path: pathlib.Path, prefix: str, n_obs: int) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    for i, scene in enumerate(scenes):
        dyn_traj = (
            np.stack(scene["obstacle_trajectories"], axis=1).astype(np.float32)
            if scene["obstacle_trajectories"]
            else np.zeros((T, 0, 4), dtype=np.float32)
        )
        actual_n_obs = int(dyn_traj.shape[1])
        if actual_n_obs != int(n_obs):
            raise ValueError(f"expected obs{n_obs}, got {actual_n_obs} in {prefix}_{i:03d}")

        dyn_seeds = np.array([o["seed"] for o in scene["obstacles"]], dtype=np.int32)
        metadata = LongInteractionSceneGenerator.build_metadata(scene)
        np.savez_compressed(
            base_path / f"{prefix}_obs{n_obs}_{i:03d}.npz",
            grid=scene["grid"],
            init_pos=scene["usv_start"],
            init_psi=np.float32(scene["usv_psi"]),
            goal=scene["usv_goal"],
            dyn_traj=dyn_traj,
            dyn_seeds=dyn_seeds,
            **metadata,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-interaction USV scenarios")
    parser.add_argument("--n-obs", type=int, default=3)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/generator_debug/debug3_new_map")
    parser.add_argument("--prefix", type=str, default="debug3_new")
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    gen = LongInteractionSceneGenerator(n_obs=args.n_obs)
    scenes = gen.generate_scenes(args.count)
    save_scenes_new(scenes, pathlib.Path(args.output_dir), args.prefix, args.n_obs)
    print(f"Saved {len(scenes)} obs{args.n_obs} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()
