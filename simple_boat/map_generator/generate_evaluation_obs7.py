"""Standard T1024 obs7 root scene generation."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

from simple_boat.map_generator.Generate_training_new import (
    DT,
    LEARNED_ARRIVAL_STEPS,
    _distance_to_boundary,
    _inside_domain,
    _unit_from_angle,
)
from simple_boat.map_generator.obstacle_generation_common import (
    ProposalBudgetExceeded,
    continuous_min_distance,
    validate_obstacle_trajectory_counts,
)
from simple_boat.map_generator.evaluation_scene import (
    BASE_OBSTACLE_COUNT,
    HORIZON,
    MIN_PAIRWISE_DISTANCE,
    SERIALIZED_BASE_LAYER,
    PairedT1024SceneGeneratorBase,
)


STANDARD_FAMILY_SPECS = (
    {
        "family": "early_crossing_slow_wall",
        "kind": "slow_wall",
        "speed_range": (0.35, 2.5),
        "route_progress_range": (0.30, 0.45),
        "lateral_window": (0.0, 2.5),
    },
    {
        "family": "normal_headon",
        "kind": "single_headon",
        "speed_range": (0.8, 2.5),
        "route_progress_range": (0.40, 0.60),
        "lateral_window": (0.0, 2.5),
    },
    {
        "family": "late_crossing_slow_wall",
        "kind": "slow_wall",
        "speed_range": (0.35, 2.5),
        "route_progress_range": (0.55, 0.75),
        "lateral_window": (0.0, 2.5),
    },
    {
        "family": "speed_matched_decoy",
        "kind": "speed_matched",
        "speed_range": (0.6, 2.2),
        "route_progress_range": (0.35, 0.70),
        "lateral_window": (3.0, 6.0),
    },
)


class StandardObs7T1024Generator(PairedT1024SceneGeneratorBase):
    """Generate one standard seven-obstacle scene."""

    @staticmethod
    def family_for_index(index: int) -> dict:
        return dict(STANDARD_FAMILY_SPECS[index % len(STANDARD_FAMILY_SPECS)])

    def _sample_standard_candidate(
        self, scene: dict, spec: dict, obs_idx: int
    ) -> tuple[dict, np.ndarray] | None:
        self._consume_proposal()
        start = np.asarray(scene["usv_start"], dtype=np.float64)
        goal = np.asarray(scene["usv_goal"], dtype=np.float64)
        route = goal - start
        route_length = float(np.linalg.norm(route))
        if route_length <= 1e-6:
            return self._reject("route_length_failure")
        route_direction = route / route_length
        route_normal = np.array(
            [-route_direction[1], route_direction[0]], dtype=np.float64
        )

        progress = random.uniform(*spec["route_progress_range"])
        lateral = random.uniform(*spec["lateral_window"])
        side = random.choice((-1, 1))
        conflict_position = (
            start + progress * route + side * lateral * route_normal
        )
        conflict_step = int(round(progress * LEARNED_ARRIVAL_STEPS))
        if not _inside_domain(
            float(conflict_position[0]), float(conflict_position[1])
        ):
            return self._reject("boundary_failure")

        heading = self._heading_for_added_kind(
            spec["kind"], float(scene["usv_psi"]), route_direction
        )
        direction = _unit_from_angle(heading).astype(np.float64)
        travel_time = (conflict_step + 1) * DT
        maximum_speed = (
            _distance_to_boundary(conflict_position, -direction)
            / travel_time
        )
        speed_low, speed_high = spec["speed_range"]
        feasible_high = min(float(speed_high), float(maximum_speed))
        if feasible_high < float(speed_low):
            return self._reject("speed_range_infeasible")
        speed = (
            float(speed_low)
            if math.isclose(feasible_high, float(speed_low))
            else random.uniform(float(speed_low), feasible_high)
        )
        obstacle_start = conflict_position - speed * travel_time * direction
        if not _inside_domain(
            float(obstacle_start[0]), float(obstacle_start[1])
        ):
            return self._reject("boundary_failure")
        if float(np.linalg.norm(obstacle_start - start)) < self.min_obs_obs_dist:
            return self._reject("initial_clearance_failure")

        obstacle_start = obstacle_start.astype(np.float32)
        trajectory = self._generate_linear_obstacle_trajectory(
            obstacle_start, speed, heading
        )
        if (
            float(
                np.linalg.norm(
                    trajectory[conflict_step, :2] - conflict_position
                )
            )
            > 0.35
        ):
            return self._reject("template_constraint_failure")
        if self._collides_with_existing(
            obstacle_start,
            trajectory,
            scene["obstacles"],
            scene["obstacle_trajectories"],
        ):
            return self._reject("pair_overlap")

        active_until = self._measure_active_until(trajectory)
        return (
            {
                "start_pos": obstacle_start,
                "velocity": speed,
                "heading": float(heading),
                "meet_time": travel_time,
                "meet_position": np.asarray(
                    conflict_position, dtype=np.float32
                ),
                "type": spec["kind"],
                "timing_role": spec["family"],
                "conflict_step": conflict_step,
                "active_until_step": int(active_until),
                "target_active_until_step": int(active_until),
                "obs_idx": int(obs_idx),
                "seed": random.randint(0, 2**31 - 1),
            },
            trajectory,
        )

    def _append_standard(self, scene: dict, spec: dict) -> bool:
        for _ in range(self.max_added_proposals):
            candidate = self._sample_standard_candidate(
                scene, spec, len(scene["obstacles"])
            )
            if candidate is None:
                continue
            scene["obstacles"].append(candidate[0])
            scene["obstacle_trajectories"].append(candidate[1])
            return True
        self._reject("augmentation_placement_failure")
        return False

    def _validate_count_scene(self, scene: dict, expected_count: int) -> float:
        obstacles = scene["obstacles"]
        trajectories = scene["obstacle_trajectories"]
        validate_obstacle_trajectory_counts(obstacles, trajectories)
        if len(obstacles) != expected_count:
            raise ValueError(
                f"expected {expected_count} obstacles, got {len(obstacles)}"
            )
        dynamic = np.stack(trajectories, axis=1)
        if dynamic.shape != (HORIZON, expected_count, 4):
            raise ValueError(f"invalid dynamic trajectory shape: {dynamic.shape}")
        starts = np.stack([obstacle["start_pos"] for obstacle in obstacles])
        if not (np.isfinite(starts).all() and np.isfinite(dynamic).all()):
            raise ValueError("scene contains non-finite values")
        centers = dynamic[:, :, :2]
        if any(
            not _inside_domain(float(x), float(y))
            for x, y in centers.reshape(-1, 2)
        ):
            raise ValueError("obstacle center outside domain")
        if np.any(
            np.linalg.norm(starts - np.asarray(scene["usv_start"]), axis=1)
            < self.min_obs_obs_dist
        ):
            raise ValueError("obstacle overlaps ownship at time zero")
        minimum = self._scene_min_pairwise_distance(scene)
        if minimum < MIN_PAIRWISE_DISTANCE - 1e-6:
            raise ValueError(f"pairwise clearance below 2m: {minimum}")
        return float(minimum)

    def generate_scene(self, index: int) -> dict:
        spec = self.family_for_index(index)
        self._proposal_count = 0
        self._rejection_counts.clear()
        attempts = 0
        for attempt in range(self.max_scene_attempts):
            attempts = attempt + 1
            seed = (
                self.master_seed + index * 1_000_003 + attempt
            ) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            try:
                prefix = self._generate_prefix_scene()
                if prefix is None:
                    self._reject("prefix_failure")
                    continue
                obs7 = copy.deepcopy(prefix)
                if not self._append_standard(obs7, spec):
                    self._reject("augmentation_failure")
                    continue
            except ProposalBudgetExceeded:
                self._reject("proposal_budget_exceeded")
                break

            try:
                obs7_minimum = self._validate_count_scene(
                    obs7, BASE_OBSTACLE_COUNT + 1
                )
            except ValueError:
                self._reject("final_validation_failure")
                continue

            base_map_id = f"paired_standard_{index:03d}"
            shared = {
                "benchmark_class": "count_generalization_in_distribution",
                "base_map_id": base_map_id,
                "augmentation_family": spec["family"],
                "conflict_route_progress": float(
                    np.dot(
                        obs7["obstacles"][6]["meet_position"]
                        - np.asarray(obs7["usv_start"]),
                        np.asarray(obs7["usv_goal"])
                        - np.asarray(obs7["usv_start"]),
                    )
                    / np.dot(
                        np.asarray(obs7["usv_goal"])
                        - np.asarray(obs7["usv_start"]),
                        np.asarray(obs7["usv_goal"])
                        - np.asarray(obs7["usv_start"]),
                    )
                ),
                "generation_seed": int(seed),
            }
            obs7.update(shared)
            obs7["source_layers"] = [SERIALIZED_BASE_LAYER] * 6 + [
                "augmentation"
            ]
            obs7["min_pairwise_distance"] = obs7_minimum
            return obs7

        failures = dict(sorted(self._rejection_counts.items()))
        raise RuntimeError(
            f"scene_index={index} family={spec['family']} attempts={attempts} "
            f"proposals={self._proposal_count} failures={failures}"
        )

    @staticmethod
    def _scene_arrays(scene: dict, expected_count: int) -> dict[str, np.ndarray]:
        obstacles = scene["obstacles"]
        trajectories = scene["obstacle_trajectories"]
        if len(obstacles) != expected_count or len(trajectories) != expected_count:
            raise ValueError(
                f"expected {expected_count} obstacles and trajectories"
            )
        metadata = StandardObs7T1024Generator.build_metadata(scene)
        return {
            "grid": np.asarray(scene["grid"], dtype=np.uint8),
            "init_pos": np.asarray(scene["usv_start"], dtype=np.float32),
            "init_psi": np.asarray(scene["usv_psi"], dtype=np.float32),
            "goal": np.asarray(scene["usv_goal"], dtype=np.float32),
            "dyn_traj": np.stack(trajectories, axis=1).astype(np.float32),
            "dyn_start_pos": np.stack(
                [obstacle["start_pos"] for obstacle in obstacles]
            ).astype(np.float32),
            "dyn_seeds": np.asarray(
                [obstacle["seed"] for obstacle in obstacles], dtype=np.int32
            ),
            **metadata,
            "source_layers": np.asarray(scene["source_layers"], dtype="U16"),
            "benchmark_class": np.asarray(
                scene["benchmark_class"], dtype="U48"
            ),
            "base_map_id": np.asarray(scene["base_map_id"], dtype="U64"),
            "augmentation_family": np.asarray(
                scene["augmentation_family"], dtype="U32"
            ),
            "conflict_route_progress": np.asarray(
                scene["conflict_route_progress"], dtype=np.float32
            ),
            "generation_seed": np.asarray(
                scene["generation_seed"], dtype=np.int64
            ),
            "min_pairwise_distance": np.asarray(
                scene["min_pairwise_distance"], dtype=np.float32
            ),
        }

    @classmethod
    def _validate_saved_file(
        cls, path: str | os.PathLike, expected_count: int
    ) -> None:
        required = {
            "grid": ((32, 32), np.dtype(np.uint8)),
            "init_pos": ((2,), np.dtype(np.float32)),
            "init_psi": ((), np.dtype(np.float32)),
            "goal": ((2,), np.dtype(np.float32)),
            "dyn_traj": (
                (HORIZON, expected_count, 4),
                np.dtype(np.float32),
            ),
            "dyn_start_pos": ((expected_count, 2), np.dtype(np.float32)),
            "dyn_seeds": ((expected_count,), np.dtype(np.int32)),
            "obs_types": ((expected_count,), np.dtype("U32")),
            "timing_roles": ((expected_count,), np.dtype("U32")),
            "conflict_steps": ((expected_count,), np.dtype(np.int32)),
            "active_until_steps": (
                (expected_count,),
                np.dtype(np.int32),
            ),
            "target_active_until_steps": (
                (expected_count,),
                np.dtype(np.int32),
            ),
            "source_layers": ((expected_count,), np.dtype("U16")),
            "benchmark_class": ((), np.dtype("U48")),
            "base_map_id": ((), np.dtype("U64")),
            "augmentation_family": ((), np.dtype("U32")),
            "conflict_route_progress": ((), np.dtype(np.float32)),
            "generation_seed": ((), np.dtype(np.int64)),
            "min_pairwise_distance": ((), np.dtype(np.float32)),
        }
        try:
            with np.load(path, allow_pickle=False) as saved:
                missing = required.keys() - saved.files
                if missing:
                    raise ValueError(f"missing required keys: {sorted(missing)}")
                arrays = {key: saved[key] for key in required}
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"invalid NPZ file {path}: {error}") from error

        for key, (shape, dtype) in required.items():
            if arrays[key].shape != shape:
                raise ValueError(
                    f"{key} has shape {arrays[key].shape}, expected {shape}"
                )
            if arrays[key].dtype != dtype:
                raise ValueError(
                    f"{key} has dtype {arrays[key].dtype}, expected {dtype}"
                )
        numeric = (
            "init_pos",
            "init_psi",
            "goal",
            "dyn_traj",
            "dyn_start_pos",
            "dyn_seeds",
            "conflict_steps",
            "active_until_steps",
            "target_active_until_steps",
            "conflict_route_progress",
            "generation_seed",
            "min_pairwise_distance",
        )
        if not all(np.isfinite(arrays[key]).all() for key in numeric):
            raise ValueError("saved scene contains non-finite values")
        if str(arrays["benchmark_class"].item()) != (
            "count_generalization_in_distribution"
        ):
            raise ValueError("invalid benchmark_class")
        family = str(arrays["augmentation_family"].item())
        specs = {spec["family"]: spec for spec in STANDARD_FAMILY_SPECS}
        if family not in specs:
            raise ValueError(f"invalid augmentation_family: {family}")
        progress = float(arrays["conflict_route_progress"])
        progress_low, progress_high = specs[family]["route_progress_range"]
        if not progress_low - 1e-6 <= progress <= progress_high + 1e-6:
            raise ValueError("conflict_route_progress outside family range")

        expected_layers = [SERIALIZED_BASE_LAYER] * BASE_OBSTACLE_COUNT
        if expected_count == BASE_OBSTACLE_COUNT + 1:
            expected_layers.append("augmentation")
            speed = float(np.linalg.norm(arrays["dyn_traj"][0, 6, 2:4]))
            speed_low, speed_high = specs[family]["speed_range"]
            if not speed_low - 1e-6 <= speed <= speed_high + 1e-6:
                raise ValueError("augmentation speed outside family range")
        if list(arrays["source_layers"].astype(str)) != expected_layers:
            raise ValueError("invalid source_layers")

        starts = arrays["dyn_start_pos"]
        centers = arrays["dyn_traj"][:, :, :2]
        velocities = arrays["dyn_traj"][:, :, 2:4]
        if any(
            not _inside_domain(float(x), float(y))
            for x, y in centers.reshape(-1, 2)
        ):
            raise ValueError("obstacle center outside domain")
        if np.any(
            np.linalg.norm(starts - arrays["init_pos"], axis=1)
            < MIN_PAIRWISE_DISTANCE
        ):
            raise ValueError("obstacle overlaps ownship at time zero")

        for index in range(expected_count):
            positions = centers[:, index]
            speed = np.linalg.norm(velocities[:, index], axis=1)
            active = speed > 1e-4
            active_count = int(np.count_nonzero(active))
            if active_count != int(arrays["active_until_steps"][index]):
                raise ValueError(f"obstacle {index} active count mismatch")
            if active_count:
                if not active[:active_count].all() or active[active_count:].any():
                    raise ValueError(f"obstacle {index} active interval mismatch")
                if not np.allclose(
                    velocities[:active_count, index],
                    velocities[0, index],
                    atol=2e-5,
                ):
                    raise ValueError(f"obstacle {index} velocity changed")
                previous = np.concatenate(
                    (starts[index : index + 1], positions[:-1]), axis=0
                )
                if not np.allclose(
                    positions[:active_count] - previous[:active_count],
                    velocities[:active_count, index] * DT,
                    atol=2e-5,
                ):
                    raise ValueError(f"obstacle {index} integration mismatch")
                if not np.allclose(
                    positions[active_count:],
                    positions[active_count - 1],
                    atol=2e-5,
                ):
                    raise ValueError(f"obstacle {index} boundary stop mismatch")

        positions = [
            np.concatenate((starts[i : i + 1], centers[:, i]), axis=0)
            for i in range(expected_count)
        ]
        minimum = min(
            continuous_min_distance(positions[first], positions[second])
            for first, second in combinations(range(expected_count), 2)
        )
        if minimum < MIN_PAIRWISE_DISTANCE - 1e-6:
            raise ValueError(f"pairwise clearance below 2m: {minimum}")
        if not math.isclose(
            minimum,
            float(arrays["min_pairwise_distance"]),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("recorded minimum pairwise distance mismatch")

    def save_scene(
        self,
        scene: dict,
        *,
        output_dir: str | os.PathLike,
        prefix: str,
        index: int,
    ) -> Path:
        if "/" in prefix or "\\" in prefix or Path(prefix).drive:
            raise ValueError("prefix must be a filename without separators")
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{prefix}_obs7_{index:03d}.npz"
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing file: {destination}"
            )

        arrays = self._scene_arrays(scene, BASE_OBSTACLE_COUNT + 1)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                np.savez_compressed(file, **arrays)
                file.flush()
                os.fsync(file.fileno())
            self._validate_saved_file(temp_path, BASE_OBSTACLE_COUNT + 1)
            os.link(temp_path, destination)
            return destination
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate standard obs7 T1024 scenarios"
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default=(
            "simple_boat/assets/"
            "eval7_maps"
        ),
    )
    parser.add_argument(
        "--prefix", default="eval7_paired_standard_t1024"
    )
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")

    generator = StandardObs7T1024Generator(master_seed=args.seed)
    for index in range(args.count):
        generator.save_scene(
            generator.generate_scene(index),
            output_dir=args.output_dir,
            prefix=args.prefix,
            index=index,
        )
    print(
        f"Saved {args.count} standard T1024 obs7 scenes to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
