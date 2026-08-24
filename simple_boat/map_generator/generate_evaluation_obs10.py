"""Build standard obs10 scenes by appending one obstacle to saved obs9 maps."""

from __future__ import annotations

import argparse
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
from simple_boat.map_generator.evaluation_scene import (
    HORIZON,
    MIN_PAIRWISE_DISTANCE,
)
from simple_boat.map_generator.generate_evaluation_obs7 import (
    STANDARD_FAMILY_SPECS,
)
from simple_boat.map_generator.generate_evaluation_obs8 import (
    FALLBACK_SEED_STRIDE,
    FORMAL_COUNT,
)
from simple_boat.map_generator.generate_evaluation_obs9 import (
    Obs9StandardFromObs8Generator,
)


OBS10_FALLBACK_LATERAL_HIGHS = (3.0, 4.0, 5.0) + (6.0,) * 21


class Obs10StandardFromObs9Generator(Obs9StandardFromObs8Generator):
    """Append one deterministic standard-family obstacle to an obs9 parent."""

    @staticmethod
    def family_quadruplet_for_index(index: int) -> dict[str, str]:
        if not 0 <= index < FORMAL_COUNT:
            raise ValueError(
                f"index must be in [0, {FORMAL_COUNT}), got {index}"
            )
        prior = Obs9StandardFromObs8Generator.family_triplet_for_index(index)
        names = tuple(spec["family"] for spec in STANDARD_FAMILY_SPECS)
        name_to_index = {name: i for i, name in enumerate(names)}
        triplet = tuple(
            name_to_index[prior[f"family_{stage}"]]
            for stage in (7, 8, 9)
        )
        occurrence = index // 64 if index < 192 else 3
        omitted = (triplet[0] + triplet[1] + 3 * triplet[2]) % 4
        family10_index = (
            omitted
            if occurrence == 3
            else (omitted + (1, 3, 2)[occurrence]) % 4
        )
        return {**prior, "family_10": names[family10_index]}

    def _load_obs9_parent(
        self,
        parent_path: str | os.PathLike,
        obs8_parent_path: str | os.PathLike,
        obs7_parent_path: str | os.PathLike,
        index: int,
    ) -> tuple[dict[str, np.ndarray], dict]:
        parent_path = Path(parent_path)
        Obs9StandardFromObs8Generator.validate_saved_file(
            parent_path, obs8_parent_path, obs7_parent_path
        )
        with np.load(parent_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        expected = self.family_quadruplet_for_index(index)
        if str(arrays["base_map_id"].item()) != f"paired_standard_{index:03d}":
            raise ValueError("parent base_map_id mismatch")
        for stage in (7, 8, 9):
            actual = str(arrays[f"augmentation_family_{stage}"].item())
            if actual != expected[f"family_{stage}"]:
                raise ValueError(
                    f"parent family_{stage} mismatch: expected "
                    f"{expected[f'family_{stage}']}, got {actual}"
                )
        scene = {
            "grid": arrays["grid"].copy(),
            "usv_start": arrays["init_pos"].copy(),
            "usv_psi": float(arrays["init_psi"]),
            "usv_goal": arrays["goal"].copy(),
            "obstacles": [
                {"start_pos": arrays["dyn_start_pos"][i].copy()}
                for i in range(9)
            ],
            "obstacle_trajectories": [
                arrays["dyn_traj"][:, i].copy() for i in range(9)
            ],
        }
        return arrays, scene

    def generate_from_parent(
        self,
        parent_path: str | os.PathLike,
        obs8_parent_path: str | os.PathLike,
        obs7_parent_path: str | os.PathLike,
        index: int,
    ) -> dict:
        parent_path = Path(parent_path)
        parent_arrays, scene = self._load_obs9_parent(
            parent_path, obs8_parent_path, obs7_parent_path, index
        )
        quadruplet = self.family_quadruplet_for_index(index)
        family10_spec = next(
            spec
            for spec in STANDARD_FAMILY_SPECS
            if spec["family"] == quadruplet["family_10"]
        )
        seed = (self.master_seed + 10_000_019 + index * 1_000_003) % (
            2**32
        )
        random.seed(seed)
        np.random.seed(seed)
        self._proposal_count = 0
        self._rejection_counts.clear()
        fallback_stage = 0
        fallback_lateral_high = float(family10_spec["lateral_window"][1])
        try:
            placed = self._append_standard(scene, family10_spec)
            allowed = {
                "augmentation_placement_failure",
                "initial_clearance_failure",
                "pair_overlap",
                "speed_range_infeasible",
            }
            relevant_failures = sum(
                self._rejection_counts.get(reason, 0)
                for reason in (
                    "pair_overlap",
                    "speed_range_infeasible",
                    "initial_clearance_failure",
                )
            )
            if (
                not placed
                and relevant_failures
                and not (set(self._rejection_counts) - allowed)
            ):
                for fallback_stage, lateral_high in enumerate(
                    OBS10_FALLBACK_LATERAL_HIGHS, start=1
                ):
                    stage_lateral_high = max(
                        float(family10_spec["lateral_window"][1]),
                        float(lateral_high),
                    )
                    stage_seed = (
                        seed + fallback_stage * FALLBACK_SEED_STRIDE
                    ) % (2**32)
                    random.seed(stage_seed)
                    np.random.seed(stage_seed)
                    stage_spec = dict(family10_spec)
                    stage_spec["lateral_window"] = (
                        family10_spec["lateral_window"][0],
                        stage_lateral_high,
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
                        fallback_lateral_high = stage_lateral_high
                        break
                    if placed:
                        break
        except ProposalBudgetExceeded:
            self._reject("proposal_budget_exceeded")
            placed = False
        if not placed:
            failures = dict(sorted(self._rejection_counts.items()))
            raise RuntimeError(
                f"scene_index={index} "
                + " ".join(
                    f"family_{stage}={quadruplet[f'family_{stage}']}"
                    for stage in (7, 8, 9, 10)
                )
                + f" proposals={self._proposal_count}/{self.max_proposals} "
                f"failures={failures}"
            )

        minimum = self._validate_count_scene(scene, 10)
        obstacle = scene["obstacles"][9]
        progress, lateral = self._conflict_geometry(scene, obstacle)
        return {
            "parent_arrays": parent_arrays,
            "parent_file": parent_path.name,
            "parent_sha256": self._sha256(parent_path),
            **quadruplet,
            "tenth_obstacle": obstacle,
            "tenth_trajectory": scene["obstacle_trajectories"][9],
            "conflict_route_progress_10": progress,
            "conflict_lateral_offset_10": lateral,
            "fallback_lateral_high_10": fallback_lateral_high,
            "fallback_stage_10": int(fallback_stage),
            "generation_seed_10": int(seed),
            "generation_proposals_10": int(self._proposal_count),
            "min_pairwise_distance": float(minimum),
            "rejection_counts": dict(sorted(self._rejection_counts.items())),
        }

    @staticmethod
    def _scene_arrays(generated: dict) -> dict[str, np.ndarray]:
        arrays = {
            key: value.copy()
            for key, value in generated["parent_arrays"].items()
        }
        obstacle = generated["tenth_obstacle"]
        arrays.update(
            {
                "dyn_traj": np.concatenate(
                    (
                        arrays["dyn_traj"],
                        np.asarray(
                            generated["tenth_trajectory"], dtype=np.float32
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
                "parent_obs9_file": np.asarray(
                    generated["parent_file"], dtype="U128"
                ),
                "parent_obs9_sha256": np.asarray(
                    generated["parent_sha256"], dtype="U64"
                ),
                "augmentation_family_10": np.asarray(
                    generated["family_10"], dtype="U32"
                ),
                "conflict_route_progress_10": np.asarray(
                    generated["conflict_route_progress_10"], dtype=np.float32
                ),
                "conflict_lateral_offset_10": np.asarray(
                    generated["conflict_lateral_offset_10"], dtype=np.float32
                ),
                "fallback_lateral_high_10": np.asarray(
                    generated["fallback_lateral_high_10"], dtype=np.float32
                ),
                "fallback_stage_10": np.asarray(
                    generated["fallback_stage_10"], dtype=np.int32
                ),
                "generation_seed_10": np.asarray(
                    generated["generation_seed_10"], dtype=np.int64
                ),
                "generation_proposals_10": np.asarray(
                    generated["generation_proposals_10"], dtype=np.int32
                ),
                "augmentation_stage": np.asarray(10, dtype=np.int32),
                "min_pairwise_distance": np.asarray(
                    generated["min_pairwise_distance"], dtype=np.float32
                ),
            }
        )
        return arrays

    @classmethod
    def validate_saved_file(
        cls,
        path: str | os.PathLike,
        parent_path: str | os.PathLike,
        obs8_parent_path: str | os.PathLike,
        obs7_parent_path: str | os.PathLike,
    ) -> None:
        path, parent_path = Path(path), Path(parent_path)
        Obs9StandardFromObs8Generator.validate_saved_file(
            parent_path, obs8_parent_path, obs7_parent_path
        )
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            with np.load(parent_path, allow_pickle=False) as saved:
                parent = {key: saved[key] for key in saved.files}
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"invalid NPZ file {path}: {error}") from error

        vector_shapes = {
            "dyn_traj": (HORIZON, 10, 4),
            "dyn_start_pos": (10, 2),
            "dyn_seeds": (10,),
            "obs_types": (10,),
            "timing_roles": (10,),
            "conflict_steps": (10,),
            "active_until_steps": (10,),
            "target_active_until_steps": (10,),
            "source_layers": (10,),
        }
        vector_keys = set(vector_shapes)
        for key, shape in vector_shapes.items():
            if key not in arrays or arrays[key].shape != shape:
                raise ValueError(f"{key} has invalid shape")
        required = {
            "parent_obs9_file": ((), np.dtype("U128")),
            "parent_obs9_sha256": ((), np.dtype("U64")),
            "augmentation_family_10": ((), np.dtype("U32")),
            "conflict_route_progress_10": ((), np.dtype(np.float32)),
            "conflict_lateral_offset_10": ((), np.dtype(np.float32)),
            "fallback_lateral_high_10": ((), np.dtype(np.float32)),
            "fallback_stage_10": ((), np.dtype(np.int32)),
            "generation_seed_10": ((), np.dtype(np.int64)),
            "generation_proposals_10": ((), np.dtype(np.int32)),
            "augmentation_stage": ((), np.dtype(np.int32)),
            "min_pairwise_distance": ((), np.dtype(np.float32)),
        }
        for key, (shape, dtype) in required.items():
            if key not in arrays:
                raise ValueError(f"missing required key: {key}")
            if arrays[key].shape != shape or arrays[key].dtype != dtype:
                raise ValueError(f"{key} has invalid shape or dtype")

        for key, value in parent.items():
            if key in vector_keys:
                prefix = (
                    arrays[key][:, :9]
                    if key == "dyn_traj"
                    else arrays[key][:9]
                )
                if not np.array_equal(prefix, value):
                    raise ValueError(f"parent {key} prefix mismatch")
            elif key not in {"augmentation_stage", "min_pairwise_distance"}:
                if key not in arrays or not np.array_equal(arrays[key], value):
                    raise ValueError(f"parent {key} mismatch")
        if str(arrays["parent_obs9_file"].item()) != parent_path.name:
            raise ValueError("parent_obs9_file mismatch")
        if str(arrays["parent_obs9_sha256"].item()) != cls._sha256(parent_path):
            raise ValueError("parent_obs9_sha256 mismatch")

        numeric = (
            "dyn_traj",
            "dyn_start_pos",
            "dyn_seeds",
            "conflict_steps",
            "active_until_steps",
            "target_active_until_steps",
            "conflict_route_progress_10",
            "conflict_lateral_offset_10",
            "fallback_lateral_high_10",
            "fallback_stage_10",
            "generation_seed_10",
            "generation_proposals_10",
            "augmentation_stage",
            "min_pairwise_distance",
        )
        if not all(np.isfinite(arrays[key]).all() for key in numeric):
            raise ValueError("saved scene contains non-finite values")

        try:
            index = int(
                str(arrays["base_map_id"].item()).removeprefix(
                    "paired_standard_"
                )
            )
        except ValueError as error:
            raise ValueError("invalid base_map_id") from error
        expected = cls.family_quadruplet_for_index(index)
        for stage in (7, 8, 9, 10):
            actual = str(arrays[f"augmentation_family_{stage}"].item())
            if actual != expected[f"family_{stage}"]:
                raise ValueError(f"scheduled family_{stage} mismatch")
        if int(arrays["augmentation_stage"].item()) != 10:
            raise ValueError("augmentation_stage must be 10")
        if str(arrays["source_layers"][9]) != "augmentation":
            raise ValueError("tenth source layer must be augmentation")

        specs = {spec["family"]: spec for spec in STANDARD_FAMILY_SPECS}
        family10 = str(arrays["augmentation_family_10"].item())
        spec = specs[family10]
        if str(arrays["obs_types"][9]) != spec["kind"]:
            raise ValueError("tenth obs_type does not match family_10")
        if str(arrays["timing_roles"][9]) != family10:
            raise ValueError("tenth timing_role does not match family_10")
        progress = float(arrays["conflict_route_progress_10"])
        if not spec["route_progress_range"][0] - 1e-6 <= progress <= (
            spec["route_progress_range"][1] + 1e-6
        ):
            raise ValueError("conflict_route_progress_10 outside family range")
        fallback_stage = int(arrays["fallback_stage_10"])
        fallback_high = float(arrays["fallback_lateral_high_10"])
        expected_high = (
            float(spec["lateral_window"][1])
            if fallback_stage == 0
            else (
                max(
                    float(spec["lateral_window"][1]),
                    OBS10_FALLBACK_LATERAL_HIGHS[fallback_stage - 1],
                )
                if 1 <= fallback_stage <= len(OBS10_FALLBACK_LATERAL_HIGHS)
                else None
            )
        )
        if expected_high is None or not math.isclose(
            fallback_high, expected_high, abs_tol=1e-6
        ):
            raise ValueError("invalid fallback lateral stage")
        lateral = float(arrays["conflict_lateral_offset_10"])
        if not spec["lateral_window"][0] - 1e-5 <= lateral <= (
            fallback_high + 1e-5
        ):
            raise ValueError("conflict_lateral_offset_10 outside stage range")
        proposals = int(arrays["generation_proposals_10"])
        proposal_low = 1 if fallback_stage == 0 else 301 + 600 * (
            fallback_stage - 1
        )
        proposal_high = 300 + 600 * fallback_stage
        if not proposal_low <= proposals <= proposal_high:
            raise ValueError("generation_proposals_10 outside bounded budget")

        starts = arrays["dyn_start_pos"]
        centers = arrays["dyn_traj"][:, :, :2]
        velocities = arrays["dyn_traj"][:, :, 2:4]
        if any(
            not _inside_domain(float(x), float(y))
            for x, y in np.concatenate(
                (starts[9:10], centers[:, 9].reshape(-1, 2))
            )
        ):
            raise ValueError("tenth obstacle center outside domain")
        if np.linalg.norm(starts[9] - arrays["init_pos"]) < MIN_PAIRWISE_DISTANCE:
            raise ValueError("tenth obstacle overlaps ownship at time zero")

        route = arrays["goal"].astype(np.float64) - arrays["init_pos"].astype(
            np.float64
        )
        route_normal = np.array([-route[1], route[0]]) / np.linalg.norm(route)
        conflict_step = int(arrays["conflict_steps"][9])
        if not 0 <= conflict_step < HORIZON:
            raise ValueError("tenth conflict_step outside horizon")
        conflict_position = centers[conflict_step, 9].astype(np.float64)
        relative = conflict_position - arrays["init_pos"].astype(np.float64)
        measured_progress = float(np.dot(relative, route) / np.dot(route, route))
        measured_lateral = abs(float(np.dot(relative, route_normal)))
        if not math.isclose(progress, measured_progress, abs_tol=1e-5):
            raise ValueError("recorded conflict route progress mismatch")
        if not math.isclose(lateral, measured_lateral, abs_tol=1e-5):
            raise ValueError("recorded conflict lateral offset mismatch")

        positions = centers[:, 9]
        tenth_velocities = velocities[:, 9]
        active = np.linalg.norm(tenth_velocities, axis=1) > 1e-4
        active_count = int(np.count_nonzero(active))
        if active_count != int(arrays["active_until_steps"][9]):
            raise ValueError("tenth active count mismatch")
        previous = np.concatenate((starts[9:10], positions[:-1]), axis=0)
        valid_motion = (
            active_count > 0
            and active[:active_count].all()
            and not active[active_count:].any()
            and np.allclose(
                tenth_velocities[:active_count],
                tenth_velocities[0],
                atol=2e-5,
            )
            and np.allclose(
                positions[:active_count] - previous[:active_count],
                tenth_velocities[:active_count] * DT,
                atol=2e-5,
            )
            and np.allclose(
                positions[active_count:], positions[active_count - 1], atol=2e-5
            )
        )
        if not valid_motion:
            raise ValueError("tenth obstacle violates constant-motion contract")
        speed = float(np.linalg.norm(tenth_velocities[0]))
        if not spec["speed_range"][0] - 1e-6 <= speed <= (
            spec["speed_range"][1] + 1e-6
        ):
            raise ValueError("augmentation speed outside family range")

        trajectories = [
            np.concatenate((starts[i : i + 1], centers[:, i]), axis=0)
            for i in range(10)
        ]
        minimum = min(
            continuous_min_distance(trajectories[first], trajectories[second])
            for first, second in combinations(range(10), 2)
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
        obs8_parent_path: str | os.PathLike,
        obs7_parent_path: str | os.PathLike,
        output_dir: str | os.PathLike,
        prefix: str,
        index: int,
    ) -> Path:
        if "/" in prefix or "\\" in prefix or Path(prefix).drive:
            raise ValueError("prefix must be a filename without separators")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{prefix}_obs10_{index:03d}.npz"
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
            self.validate_saved_file(
                temporary, parent_path, obs8_parent_path, obs7_parent_path
            )
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing file: {destination}"
                ) from error
            created = True
            self.validate_saved_file(
                destination, parent_path, obs8_parent_path, obs7_parent_path
            )
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
        description="Append one standard obstacle to saved obs9 maps"
    )
    parser.add_argument("--count", type=int, default=FORMAL_COUNT)
    parser.add_argument(
        "--obs9-parent-dir",
        default="simple_boat/assets/eval9_maps",
    )
    parser.add_argument(
        "--obs9-parent-prefix", default="eval9_paired_standard_t1024_obs9"
    )
    parser.add_argument(
        "--obs8-parent-dir",
        default="simple_boat/assets/eval8_maps",
    )
    parser.add_argument(
        "--obs8-parent-prefix", default="eval8_paired_standard_t1024_obs8"
    )
    parser.add_argument(
        "--obs7-parent-dir",
        default="simple_boat/assets/eval7_maps",
    )
    parser.add_argument(
        "--obs7-parent-prefix", default="eval7_paired_standard_t1024_obs7"
    )
    parser.add_argument(
        "--output-dir",
        default="simple_boat/assets/eval10_maps",
    )
    parser.add_argument("--prefix", default="eval10_paired_standard_t1024")
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args(argv)
    if not 0 < args.count <= FORMAL_COUNT:
        parser.error(f"--count must be in [1, {FORMAL_COUNT}]")

    obs9_dir = Path(args.obs9_parent_dir)
    obs8_dir = Path(args.obs8_parent_dir)
    obs7_dir = Path(args.obs7_parent_dir)
    output_dir = Path(args.output_dir)
    obs9_parents = [
        obs9_dir / f"{args.obs9_parent_prefix}_{index:03d}.npz"
        for index in range(args.count)
    ]
    obs8_parents = [
        obs8_dir / f"{args.obs8_parent_prefix}_{index:03d}.npz"
        for index in range(args.count)
    ]
    obs7_parents = [
        obs7_dir / f"{args.obs7_parent_prefix}_{index:03d}.npz"
        for index in range(args.count)
    ]
    missing = [
        path
        for path in (*obs9_parents, *obs8_parents, *obs7_parents)
        if not path.is_file()
    ]
    if missing:
        parser.error(f"missing parent file: {missing[0]}")
    destinations = [
        output_dir / f"{args.prefix}_obs10_{index:03d}.npz"
        for index in range(args.count)
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing file: {existing[0]}")

    generator = Obs10StandardFromObs9Generator(master_seed=args.seed)
    for index, (obs9_parent, obs8_parent, obs7_parent) in enumerate(
        zip(obs9_parents, obs8_parents, obs7_parents)
    ):
        generated = generator.generate_from_parent(
            obs9_parent, obs8_parent, obs7_parent, index
        )
        generator.save_scene(
            generated,
            parent_path=obs9_parent,
            obs8_parent_path=obs8_parent,
            obs7_parent_path=obs7_parent,
            output_dir=output_dir,
            prefix=args.prefix,
            index=index,
        )
    print(f"Saved {args.count} standard obs10 scenes to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
