"""Build standard obs8 scenes by appending one obstacle to saved obs7 maps."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np

from simple_boat.map_generator.Generate_training_new import DT, _inside_domain
from simple_boat.map_generator.obstacle_generation_common import (
    ProposalBudgetExceeded,
    continuous_min_distance,
)
from simple_boat.map_generator.generate_evaluation_obs7 import (
    StandardObs7T1024Generator,
    STANDARD_FAMILY_SPECS,
)
from simple_boat.map_generator.evaluation_scene import (
    HORIZON,
    MIN_PAIRWISE_DISTANCE,
)


FORMAL_COUNT = 200
TAIL_FAMILY8_INDICES = (0, 1, 2, 3, 1, 2, 3, 0)
FALLBACK_LATERAL_HIGHS = (3.0, 4.0, 5.0, 6.0)
FALLBACK_SEED_STRIDE = 97_409


class Obs8StandardFromObs7Generator(StandardObs7T1024Generator):
    """Append a deterministic standard-family obstacle to an obs7 parent."""

    @staticmethod
    def family_pair_for_index(index: int) -> dict[str, str]:
        if not 0 <= index < FORMAL_COUNT:
            raise ValueError(
                f"index must be in [0, {FORMAL_COUNT}), got {index}"
            )
        family7_index = index % len(STANDARD_FAMILY_SPECS)
        family8_index = (
            (index // len(STANDARD_FAMILY_SPECS))
            % len(STANDARD_FAMILY_SPECS)
            if index < 192
            else TAIL_FAMILY8_INDICES[index - 192]
        )
        return {
            "family_7": STANDARD_FAMILY_SPECS[family7_index]["family"],
            "family_8": STANDARD_FAMILY_SPECS[family8_index]["family"],
        }

    @staticmethod
    def _sha256(path: str | Path) -> str:
        return sha256(Path(path).read_bytes()).hexdigest()

    def _load_parent(
        self, parent_path: str | Path, index: int
    ) -> tuple[dict[str, np.ndarray], dict]:
        parent_path = Path(parent_path)
        with np.load(parent_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        expected = self.family_pair_for_index(index)
        base_map_id = str(arrays.get("base_map_id", np.asarray("")).item())
        if base_map_id != f"paired_standard_{index:03d}":
            raise ValueError(
                f"parent base_map_id mismatch: {base_map_id!r}"
            )
        family7 = str(
            arrays.get("augmentation_family", np.asarray("")).item()
        )
        if family7 != expected["family_7"]:
            raise ValueError(
                "parent family mismatch: "
                f"expected {expected['family_7']}, got {family7}"
            )
        self._validate_saved_file(parent_path, 7)
        scene = {
            "grid": arrays["grid"].copy(),
            "usv_start": arrays["init_pos"].copy(),
            "usv_psi": float(arrays["init_psi"]),
            "usv_goal": arrays["goal"].copy(),
            "obstacles": [
                {"start_pos": arrays["dyn_start_pos"][i].copy()}
                for i in range(7)
            ],
            "obstacle_trajectories": [
                arrays["dyn_traj"][:, i].copy() for i in range(7)
            ],
        }
        return arrays, scene

    def generate_from_parent(
        self, parent_path: str | Path, index: int
    ) -> dict:
        parent_path = Path(parent_path)
        parent_arrays, scene = self._load_parent(parent_path, index)
        pair = self.family_pair_for_index(index)
        family8_spec = next(
            spec
            for spec in STANDARD_FAMILY_SPECS
            if spec["family"] == pair["family_8"]
        )
        seed = (
            self.master_seed + 8_000_019 + index * 1_000_003
        ) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        self._proposal_count = 0
        self._rejection_counts.clear()
        try:
            placed = self._append_standard(scene, family8_spec)
            fallback_stage = 0
            fallback_lateral_high = float(
                family8_spec["lateral_window"][1]
            )
            allowed = {
                "augmentation_placement_failure",
                "pair_overlap",
                "speed_range_infeasible",
            }
            if (
                not placed
                and (
                    self._rejection_counts.get("pair_overlap", 0)
                    + self._rejection_counts.get(
                        "speed_range_infeasible", 0
                    )
                )
                and not (set(self._rejection_counts) - allowed)
            ):
                for fallback_stage, lateral_high in enumerate(
                    FALLBACK_LATERAL_HIGHS, start=1
                ):
                    if lateral_high <= family8_spec["lateral_window"][1]:
                        continue
                    stage_seed = (
                        seed + fallback_stage * FALLBACK_SEED_STRIDE
                    ) % (2**32)
                    random.seed(stage_seed)
                    np.random.seed(stage_seed)
                    stage_spec = dict(family8_spec)
                    stage_spec["lateral_window"] = (
                        family8_spec["lateral_window"][0],
                        lateral_high,
                    )
                    for _ in range(2 * self.max_added_proposals):
                        candidate = self._sample_standard_candidate(
                            scene, stage_spec, len(scene["obstacles"])
                        )
                        if candidate is None:
                            continue
                        scene["obstacles"].append(candidate[0])
                        scene["obstacle_trajectories"].append(candidate[1])
                        placed = True
                        fallback_lateral_high = float(lateral_high)
                        break
                    if placed:
                        break
        except ProposalBudgetExceeded:
            self._reject("proposal_budget_exceeded")
            placed = False
        if not placed:
            failures = dict(sorted(self._rejection_counts.items()))
            raise RuntimeError(
                f"scene_index={index} family_7={pair['family_7']} "
                f"family_8={pair['family_8']} proposals="
                f"{self._proposal_count}/{self.max_proposals} "
                f"failures={failures}"
            )

        minimum = self._validate_count_scene(scene, 8)
        eighth_obstacle = scene["obstacles"][7]
        route = np.asarray(scene["usv_goal"], dtype=np.float64) - np.asarray(
            scene["usv_start"], dtype=np.float64
        )
        progress = float(
            np.dot(
                np.asarray(eighth_obstacle["meet_position"], dtype=np.float64)
                - np.asarray(scene["usv_start"], dtype=np.float64),
                route,
            )
            / np.dot(route, route)
        )
        route_normal = np.array(
            [-route[1], route[0]], dtype=np.float64
        ) / np.linalg.norm(route)
        lateral_offset = abs(
            float(
                np.dot(
                    np.asarray(
                        eighth_obstacle["meet_position"], dtype=np.float64
                    )
                    - np.asarray(scene["usv_start"], dtype=np.float64),
                    route_normal,
                )
            )
        )
        return {
            "parent_arrays": parent_arrays,
            "parent_file": parent_path.name,
            "parent_sha256": self._sha256(parent_path),
            "family_7": pair["family_7"],
            "family_8": pair["family_8"],
            "eighth_obstacle": eighth_obstacle,
            "eighth_trajectory": scene["obstacle_trajectories"][7],
            "conflict_route_progress_8": progress,
            "conflict_lateral_offset_8": lateral_offset,
            "fallback_lateral_high_8": fallback_lateral_high,
            "fallback_stage_8": int(fallback_stage),
            "generation_seed_8": int(seed),
            "generation_proposals_8": int(self._proposal_count),
            "min_pairwise_distance": float(minimum),
            "rejection_counts": dict(sorted(self._rejection_counts.items())),
        }

    @staticmethod
    def _scene_arrays(generated: dict) -> dict[str, np.ndarray]:
        arrays = {
            key: value.copy()
            for key, value in generated["parent_arrays"].items()
        }
        obstacle = generated["eighth_obstacle"]
        arrays.update(
            {
                "dyn_traj": np.concatenate(
                    (
                        arrays["dyn_traj"],
                        np.asarray(
                            generated["eighth_trajectory"], dtype=np.float32
                        )[:, None, :],
                    ),
                    axis=1,
                ),
                "dyn_start_pos": np.concatenate(
                    (
                        arrays["dyn_start_pos"],
                        np.asarray(obstacle["start_pos"], dtype=np.float32)[
                            None, :
                        ],
                    )
                ),
                "dyn_seeds": np.append(
                    arrays["dyn_seeds"],
                    np.asarray(obstacle["seed"], dtype=np.int32),
                ).astype(np.int32),
                "obs_types": np.append(
                    arrays["obs_types"], obstacle["type"]
                ).astype("U32"),
                "timing_roles": np.append(
                    arrays["timing_roles"], obstacle["timing_role"]
                ).astype("U32"),
                "conflict_steps": np.append(
                    arrays["conflict_steps"],
                    np.asarray(obstacle["conflict_step"], dtype=np.int32),
                ).astype(np.int32),
                "active_until_steps": np.append(
                    arrays["active_until_steps"],
                    np.asarray(
                        obstacle["active_until_step"], dtype=np.int32
                    ),
                ).astype(np.int32),
                "target_active_until_steps": np.append(
                    arrays["target_active_until_steps"],
                    np.asarray(
                        obstacle["target_active_until_step"], dtype=np.int32
                    ),
                ).astype(np.int32),
                "source_layers": np.append(
                    arrays["source_layers"], "augmentation"
                ).astype("U16"),
                "parent_obs7_file": np.asarray(
                    generated["parent_file"], dtype="U128"
                ),
                "parent_obs7_sha256": np.asarray(
                    generated["parent_sha256"], dtype="U64"
                ),
                "augmentation_family_7": np.asarray(
                    generated["family_7"], dtype="U32"
                ),
                "augmentation_family_8": np.asarray(
                    generated["family_8"], dtype="U32"
                ),
                "conflict_route_progress_8": np.asarray(
                    generated["conflict_route_progress_8"], dtype=np.float32
                ),
                "conflict_lateral_offset_8": np.asarray(
                    generated["conflict_lateral_offset_8"], dtype=np.float32
                ),
                "fallback_lateral_high_8": np.asarray(
                    generated["fallback_lateral_high_8"], dtype=np.float32
                ),
                "fallback_stage_8": np.asarray(
                    generated["fallback_stage_8"], dtype=np.int32
                ),
                "generation_seed_8": np.asarray(
                    generated["generation_seed_8"], dtype=np.int64
                ),
                "generation_proposals_8": np.asarray(
                    generated["generation_proposals_8"], dtype=np.int32
                ),
                "augmentation_stage": np.asarray(8, dtype=np.int32),
                "min_pairwise_distance": np.asarray(
                    generated["min_pairwise_distance"], dtype=np.float32
                ),
            }
        )
        return arrays

    @classmethod
    def validate_saved_file(
        cls, path: str | os.PathLike, parent_path: str | os.PathLike
    ) -> None:
        parent_path = Path(parent_path)
        cls._validate_saved_file(parent_path, 7)
        required = {
            "grid": ((32, 32), np.dtype(np.uint8)),
            "init_pos": ((2,), np.dtype(np.float32)),
            "init_psi": ((), np.dtype(np.float32)),
            "goal": ((2,), np.dtype(np.float32)),
            "dyn_traj": ((HORIZON, 8, 4), np.dtype(np.float32)),
            "dyn_start_pos": ((8, 2), np.dtype(np.float32)),
            "dyn_seeds": ((8,), np.dtype(np.int32)),
            "obs_types": ((8,), np.dtype("U32")),
            "timing_roles": ((8,), np.dtype("U32")),
            "conflict_steps": ((8,), np.dtype(np.int32)),
            "active_until_steps": ((8,), np.dtype(np.int32)),
            "target_active_until_steps": ((8,), np.dtype(np.int32)),
            "source_layers": ((8,), np.dtype("U16")),
            "benchmark_class": ((), np.dtype("U48")),
            "base_map_id": ((), np.dtype("U64")),
            "augmentation_family": ((), np.dtype("U32")),
            "conflict_route_progress": ((), np.dtype(np.float32)),
            "generation_seed": ((), np.dtype(np.int64)),
            "min_pairwise_distance": ((), np.dtype(np.float32)),
            "parent_obs7_file": ((), np.dtype("U128")),
            "parent_obs7_sha256": ((), np.dtype("U64")),
            "augmentation_family_7": ((), np.dtype("U32")),
            "augmentation_family_8": ((), np.dtype("U32")),
            "conflict_route_progress_8": ((), np.dtype(np.float32)),
            "conflict_lateral_offset_8": ((), np.dtype(np.float32)),
            "fallback_lateral_high_8": ((), np.dtype(np.float32)),
            "fallback_stage_8": ((), np.dtype(np.int32)),
            "generation_seed_8": ((), np.dtype(np.int64)),
            "generation_proposals_8": ((), np.dtype(np.int32)),
            "augmentation_stage": ((), np.dtype(np.int32)),
        }
        try:
            with np.load(path, allow_pickle=False) as saved:
                missing = required.keys() - saved.files
                if missing:
                    raise ValueError(
                        f"missing required keys: {sorted(missing)}"
                    )
                arrays = {key: saved[key] for key in required}
            with np.load(parent_path, allow_pickle=False) as saved:
                parent = {key: saved[key] for key in saved.files}
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
            "conflict_route_progress_8",
            "conflict_lateral_offset_8",
            "fallback_lateral_high_8",
            "fallback_stage_8",
            "generation_seed_8",
            "generation_proposals_8",
            "augmentation_stage",
        )
        if not all(np.isfinite(arrays[key]).all() for key in numeric):
            raise ValueError("saved scene contains non-finite values")

        for key in ("grid", "init_pos", "init_psi", "goal"):
            if not np.array_equal(arrays[key], parent[key]):
                raise ValueError(f"parent {key} mismatch")
        for key in (
            "benchmark_class",
            "base_map_id",
            "augmentation_family",
            "conflict_route_progress",
            "generation_seed",
        ):
            if not np.array_equal(arrays[key], parent[key]):
                raise ValueError(f"parent {key} mismatch")
        for key in (
            "dyn_traj",
            "dyn_start_pos",
            "dyn_seeds",
            "obs_types",
            "timing_roles",
            "conflict_steps",
            "active_until_steps",
            "target_active_until_steps",
            "source_layers",
        ):
            prefix = arrays[key][:, :7] if key == "dyn_traj" else arrays[key][:7]
            if not np.array_equal(prefix, parent[key]):
                raise ValueError(f"parent {key} prefix mismatch")

        if str(arrays["parent_obs7_file"].item()) != parent_path.name:
            raise ValueError("parent_obs7_file mismatch")
        if str(arrays["parent_obs7_sha256"].item()) != cls._sha256(
            parent_path
        ):
            raise ValueError("parent_obs7_sha256 mismatch")
        family7 = str(arrays["augmentation_family_7"].item())
        if family7 != str(parent["augmentation_family"].item()):
            raise ValueError("augmentation_family_7 mismatch")
        base_map_id = str(arrays["base_map_id"].item())
        try:
            index = int(base_map_id.removeprefix("paired_standard_"))
        except ValueError as error:
            raise ValueError("invalid base_map_id") from error
        expected_pair = cls.family_pair_for_index(index)
        if family7 != expected_pair["family_7"]:
            raise ValueError("scheduled family_7 mismatch")
        if int(arrays["augmentation_stage"].item()) != 8:
            raise ValueError("augmentation_stage must be 8")
        if str(arrays["source_layers"][7]) != "augmentation":
            raise ValueError("eighth source layer must be augmentation")

        specs = {spec["family"]: spec for spec in STANDARD_FAMILY_SPECS}
        family8 = str(arrays["augmentation_family_8"].item())
        if family8 not in specs:
            raise ValueError(f"invalid augmentation_family_8: {family8}")
        if family8 != expected_pair["family_8"]:
            raise ValueError("scheduled family_8 mismatch")
        if str(arrays["obs_types"][7]) != specs[family8]["kind"]:
            raise ValueError("eighth obs_type does not match family")
        if str(arrays["timing_roles"][7]) != family8:
            raise ValueError("eighth timing_role does not match family")
        progress = float(arrays["conflict_route_progress_8"])
        low, high = specs[family8]["route_progress_range"]
        if not low - 1e-6 <= progress <= high + 1e-6:
            raise ValueError("conflict_route_progress_8 outside family range")
        fallback_stage = int(arrays["fallback_stage_8"])
        fallback_high = float(arrays["fallback_lateral_high_8"])
        expected_high = (
            float(specs[family8]["lateral_window"][1])
            if fallback_stage == 0
            else (
                FALLBACK_LATERAL_HIGHS[fallback_stage - 1]
                if 1 <= fallback_stage <= len(FALLBACK_LATERAL_HIGHS)
                else None
            )
        )
        if expected_high is None or not math.isclose(
            fallback_high, expected_high, abs_tol=1e-6
        ):
            raise ValueError("invalid fallback lateral stage")
        lateral_offset = float(arrays["conflict_lateral_offset_8"])
        lateral_low = float(specs[family8]["lateral_window"][0])
        if not lateral_low - 1e-5 <= lateral_offset <= fallback_high + 1e-5:
            raise ValueError("conflict_lateral_offset_8 outside stage range")
        proposals = int(arrays["generation_proposals_8"])
        proposal_low = 1 if fallback_stage == 0 else 301 + 600 * (
            fallback_stage - 1
        )
        proposal_high = 300 + 600 * fallback_stage
        if not proposal_low <= proposals <= proposal_high:
            raise ValueError("generation_proposals_8 outside bounded budget")

        starts = arrays["dyn_start_pos"]
        centers = arrays["dyn_traj"][:, :, :2]
        velocities = arrays["dyn_traj"][:, :, 2:4]
        if any(
            not _inside_domain(float(x), float(y))
            for x, y in np.concatenate(
                (starts.reshape(-1, 2), centers.reshape(-1, 2)), axis=0
            )
        ):
            raise ValueError("obstacle center outside domain")
        if np.any(
            np.linalg.norm(starts - arrays["init_pos"], axis=1)
            < MIN_PAIRWISE_DISTANCE
        ):
            raise ValueError("obstacle overlaps ownship at time zero")

        route = arrays["goal"].astype(np.float64) - arrays["init_pos"].astype(
            np.float64
        )
        route_normal = np.array([-route[1], route[0]]) / np.linalg.norm(route)
        conflict_step = int(arrays["conflict_steps"][7])
        if not 0 <= conflict_step < HORIZON:
            raise ValueError("eighth conflict_step outside horizon")
        conflict_position = centers[conflict_step, 7].astype(np.float64)
        measured_progress = float(
            np.dot(
                conflict_position - arrays["init_pos"].astype(np.float64),
                route,
            )
            / np.dot(route, route)
        )
        measured_lateral = abs(
            float(
                np.dot(
                    conflict_position
                    - arrays["init_pos"].astype(np.float64),
                    route_normal,
                )
            )
        )
        if not math.isclose(progress, measured_progress, abs_tol=1e-5):
            raise ValueError("recorded conflict route progress mismatch")
        if not math.isclose(
            lateral_offset, measured_lateral, abs_tol=1e-5
        ):
            raise ValueError("recorded conflict lateral offset mismatch")

        eighth_positions = centers[:, 7]
        eighth_velocities = velocities[:, 7]
        active = np.linalg.norm(eighth_velocities, axis=1) > 1e-4
        active_count = int(np.count_nonzero(active))
        if active_count != int(arrays["active_until_steps"][7]):
            raise ValueError("obstacle 7 active count mismatch")
        if active_count:
            previous = np.concatenate(
                (starts[7:8], eighth_positions[:-1]), axis=0
            )
            valid_motion = (
                active[:active_count].all()
                and not active[active_count:].any()
                and np.allclose(
                    eighth_velocities[:active_count],
                    eighth_velocities[0],
                    atol=2e-5,
                )
                and np.allclose(
                    eighth_positions[:active_count]
                    - previous[:active_count],
                    eighth_velocities[:active_count] * DT,
                    atol=2e-5,
                )
                and np.allclose(
                    eighth_positions[active_count:],
                    eighth_positions[active_count - 1],
                    atol=2e-5,
                )
            )
        else:
            valid_motion = np.allclose(
                eighth_positions, starts[7], atol=2e-5
            )
        if not valid_motion:
            raise ValueError("obstacle 7 violates constant-motion contract")
        speed = float(np.linalg.norm(eighth_velocities[0]))
        low, high = specs[family8]["speed_range"]
        if not low - 1e-6 <= speed <= high + 1e-6:
            raise ValueError("augmentation speed outside family range")

        positions = [
            np.concatenate((starts[i : i + 1], centers[:, i]), axis=0)
            for i in range(8)
        ]
        minimum = min(
            continuous_min_distance(positions[first], positions[second])
            for first, second in combinations(range(8), 2)
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
        generated: dict,
        *,
        parent_path: str | os.PathLike,
        output_dir: str | os.PathLike,
        prefix: str,
        index: int,
    ) -> Path:
        if "/" in prefix or "\\" in prefix or Path(prefix).drive:
            raise ValueError("prefix must be a filename without separators")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{prefix}_obs8_{index:03d}.npz"
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing file: {destination}"
            )

        temporary = None
        created = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                np.savez_compressed(file, **self._scene_arrays(generated))
                file.flush()
                os.fsync(file.fileno())
            self.validate_saved_file(temporary, parent_path)
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing file: {destination}"
                ) from error
            created = True
            self.validate_saved_file(destination, parent_path)
            return destination
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one standard obstacle to saved obs7 maps"
    )
    parser.add_argument("--count", type=int, default=FORMAL_COUNT)
    parser.add_argument(
        "--parent-dir",
        default="simple_boat/assets/eval7_maps",
    )
    parser.add_argument(
        "--parent-prefix", default="eval7_paired_standard_t1024_obs7"
    )
    parser.add_argument(
        "--output-dir",
        default="simple_boat/assets/eval8_maps",
    )
    parser.add_argument("--prefix", default="eval8_paired_standard_t1024")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args(argv)
    if not 0 < args.count <= FORMAL_COUNT:
        parser.error(f"--count must be in [1, {FORMAL_COUNT}]")

    parent_dir = Path(args.parent_dir)
    output_dir = Path(args.output_dir)
    parents = [
        parent_dir / f"{args.parent_prefix}_{index:03d}.npz"
        for index in range(args.count)
    ]
    missing = [path for path in parents if not path.is_file()]
    if missing:
        parser.error(f"missing parent file: {missing[0]}")
    destinations = [
        output_dir / f"{args.prefix}_obs8_{index:03d}.npz"
        for index in range(args.count)
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing file: {existing[0]}")

    generator = Obs8StandardFromObs7Generator(master_seed=args.seed)
    for index, parent in enumerate(parents):
        generated = generator.generate_from_parent(parent, index)
        generator.save_scene(
            generated,
            parent_path=parent,
            output_dir=output_dir,
            prefix=args.prefix,
            index=index,
        )
    print(f"Saved {args.count} standard obs8 scenes to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
