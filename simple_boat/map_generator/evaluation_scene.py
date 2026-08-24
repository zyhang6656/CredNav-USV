"""Shared bounded scene mechanics for the paired T1024 obstacle chain."""

from __future__ import annotations

import math
import operator
from collections import Counter
from itertools import combinations

import numpy as np

from simple_boat.map_generator.Generate_training_new import (
    DT,
    LongInteractionSceneGenerator,
    _inside_domain,
)
from simple_boat.map_generator.obstacle_generation_common import (
    ProposalBudgetExceeded,
    continuous_min_distance,
    trajectory_positions,
    validate_obstacle_trajectory_counts,
)


HORIZON = 1024
BASE_OBSTACLE_COUNT = 6
MIN_PAIRWISE_DISTANCE = 2.0
MAX_SCENE_ATTEMPTS = 100
MAX_PROPOSALS = 50_000
MAX_ADDED_PROPOSALS = 300

# Saved-map/cache compatibility: changing this value changes NPZ hashes.
SERIALIZED_BASE_LAYER = "legacy_base"


class PairedT1024SceneGeneratorBase(LongInteractionSceneGenerator):
    """Provide bounded base6 sampling and constant-motion helpers."""

    def __init__(
        self,
        *,
        master_seed: int,
        max_scene_attempts: int = MAX_SCENE_ATTEMPTS,
        max_proposals: int = MAX_PROPOSALS,
        max_added_proposals: int = MAX_ADDED_PROPOSALS,
    ):
        requested_budgets = {
            "max_scene_attempts": (max_scene_attempts, MAX_SCENE_ATTEMPTS),
            "max_proposals": (max_proposals, MAX_PROPOSALS),
            "max_added_proposals": (
                max_added_proposals,
                MAX_ADDED_PROPOSALS,
            ),
        }
        budgets = {}
        for name, (requested, hard_cap) in requested_budgets.items():
            try:
                value = operator.index(requested)
            except TypeError:
                raise ValueError(
                    f"{name} must be between 0 and {hard_cap}, got "
                    f"{requested!r}"
                ) from None
            if not 0 <= value <= hard_cap:
                raise ValueError(
                    f"{name} must be between 0 and {hard_cap}, got {value}"
                )
            budgets[name] = value

        super().__init__(n_obs=BASE_OBSTACLE_COUNT)
        self.master_seed = int(master_seed)
        self.max_scene_attempts = budgets["max_scene_attempts"]
        self.max_proposals = budgets["max_proposals"]
        self.max_added_proposals = budgets["max_added_proposals"]
        self.max_timing_tries = 1
        self.max_tries = 300
        self.min_obs_obs_dist = MIN_PAIRWISE_DISTANCE
        self._proposal_count = 0
        self._rejection_counts = Counter()

    def _consume_proposal(self) -> None:
        if self._proposal_count >= self.max_proposals:
            raise ProposalBudgetExceeded(
                f"proposal budget {self.max_proposals} exhausted"
            )
        self._proposal_count += 1

    def _reject(self, reason: str):
        self._rejection_counts[reason] += 1
        return None

    def _sample_timed_obstacle(self, *args, **kwargs):
        self._consume_proposal()
        return super()._sample_timed_obstacle(*args, **kwargs)

    def _heading_for_added_kind(
        self, kind: str, usv_psi: float, usv_dir: np.ndarray
    ) -> float:
        return self._heading_for_type(kind, usv_psi, usv_dir)

    def _generate_prefix_scene(self) -> dict | None:
        return super()._generate_one_scene()

    @staticmethod
    def _generate_linear_obstacle_trajectory(
        start_pos: np.ndarray, velocity: float, heading: float
    ) -> np.ndarray:
        trajectory = np.zeros((HORIZON, 4), dtype=np.float32)
        x, y = float(start_pos[0]), float(start_pos[1])
        psi, speed = float(heading), float(velocity)
        inside = _inside_domain(x, y)
        stopped = False
        stop_x = stop_y = 0.0
        epsilon = 1e-5
        for step in range(HORIZON):
            if stopped:
                trajectory[step] = [
                    stop_x,
                    stop_y,
                    epsilon * math.cos(psi),
                    epsilon * math.sin(psi),
                ]
                continue
            velocity_x = speed * math.cos(psi)
            velocity_y = speed * math.sin(psi)
            next_x = x + velocity_x * DT
            next_y = y + velocity_y * DT
            next_inside = _inside_domain(next_x, next_y)
            if inside and not next_inside:
                stopped = True
                stop_x, stop_y = x, y
                trajectory[step] = [
                    stop_x,
                    stop_y,
                    epsilon * math.cos(psi),
                    epsilon * math.sin(psi),
                ]
            else:
                x, y = next_x, next_y
                inside = next_inside
                trajectory[step] = [x, y, velocity_x, velocity_y]
        return trajectory

    def _collides_with_existing(
        self,
        candidate_start: np.ndarray,
        candidate_trajectory: np.ndarray,
        existing: list[dict],
        existing_trajectories: list[np.ndarray],
    ) -> bool:
        validate_obstacle_trajectory_counts(existing, existing_trajectories)
        candidate_positions = trajectory_positions(
            candidate_start, candidate_trajectory
        )
        return any(
            continuous_min_distance(
                candidate_positions,
                trajectory_positions(obstacle["start_pos"], trajectory),
            )
            < self.min_obs_obs_dist
            for obstacle, trajectory in zip(existing, existing_trajectories)
        )

    @staticmethod
    def _scene_min_pairwise_distance(scene: dict) -> float:
        obstacles = scene["obstacles"]
        trajectories = scene["obstacle_trajectories"]
        validate_obstacle_trajectory_counts(obstacles, trajectories)
        positions = [
            trajectory_positions(obstacle["start_pos"], trajectory)
            for obstacle, trajectory in zip(obstacles, trajectories)
        ]
        return min(
            (
                continuous_min_distance(first, second)
                for first, second in combinations(positions, 2)
            ),
            default=float("inf"),
        )
