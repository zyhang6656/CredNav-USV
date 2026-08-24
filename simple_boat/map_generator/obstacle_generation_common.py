"""Shared bounded obstacle-generation geometry."""

from __future__ import annotations

import numpy as np


class ProposalBudgetExceeded(RuntimeError):
    pass


def trajectory_positions(start: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    positions = np.empty((len(trajectory) + 1, 2), dtype=np.float64)
    positions[0] = np.asarray(start, dtype=np.float64)[:2]
    positions[1:] = np.asarray(trajectory, dtype=np.float64)[:, :2]
    return positions


def continuous_min_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    count = min(len(pos_a), len(pos_b))
    if count == 0:
        return float("inf")

    relative = (
        np.asarray(pos_a, dtype=np.float64)[:count]
        - np.asarray(pos_b, dtype=np.float64)[:count]
    )
    if count == 1:
        return float(np.linalg.norm(relative[0]))

    starts = relative[:-1]
    deltas = np.diff(relative, axis=0)
    denominator = np.einsum("ij,ij->i", deltas, deltas)
    fractions = np.divide(
        -np.einsum("ij,ij->i", starts, deltas),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    np.clip(fractions, 0.0, 1.0, out=fractions)
    closest = starts + fractions[:, None] * deltas
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", closest, closest))))


def validate_obstacle_trajectory_counts(obstacles, trajectories) -> None:
    if len(obstacles) != len(trajectories):
        raise ValueError(
            "obstacle/trajectory count mismatch: "
            f"{len(obstacles)} obstacles, {len(trajectories)} trajectories"
        )
