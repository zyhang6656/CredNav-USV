"""Precompute USVEnvMinimal KF replay cache for scenario datasets.

Example:
  python scripts/precompute_kf_cache.py \
    --config configs/evaluation/delay20_cov100_binned_s10.yaml \
    --scenario-dir simple_boat/assets/nav3_new_map \
    --cache-dir results/kf_cache/obs3_delay20_cov100_binned_s10_shared \
    --all-binned-candidates
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_obs3 import build_env_kwargs
from scripts.eval_helpers import load_scenario_into_env, select_single_scenario_file
from simple_boat.envs.usv_env_minimal import (
    USVEnvMinimal,
    bin_path_progress_candidate_steps,
    compute_path_progress_candidate_steps,
)


def parse_seed_spec(spec: str) -> list[int]:
    if ":" in spec:
        start_text, end_text = spec.split(":", 1)
        return list(range(int(start_text), int(end_text)))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def write_cache_audit(path: pathlib.Path, audit: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cache audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            handle.write(json.dumps(audit, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite cache audit: {path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def cache_inventory(cache_dir: pathlib.Path) -> dict:
    temporary_files = sorted(
        str(path)
        for path in cache_dir.rglob("*")
        if path.is_file() and ".tmp" in path.name
    )
    if temporary_files:
        raise RuntimeError(
            "temporary cache files remain after precompute: "
            + ", ".join(temporary_files[:5])
        )
    cache_files = sorted(path for path in cache_dir.rglob("*.npz") if path.is_file())
    return {
        "cache_file_count": len(cache_files),
        "cache_bytes": sum(path.stat().st_size for path in cache_files),
        "temporary_files": temporary_files,
    }


def validate_cache_inventory(
    inventory: dict,
    *,
    files_seen: int,
    resets: int,
    cache_hits: int,
    cache_mode: str,
    all_binned_candidates: bool,
) -> None:
    if int(inventory["cache_bytes"]) <= 0:
        raise RuntimeError("cache inventory is empty")
    if all_binned_candidates:
        if files_seen != resets:
            raise RuntimeError(
                f"cache files seen ({files_seen}) do not match resets ({resets})"
            )
        if int(inventory["cache_file_count"]) != files_seen:
            raise RuntimeError(
                "cache inventory count "
                f"({inventory['cache_file_count']}) does not match files seen ({files_seen})"
            )
    if cache_mode == "read_strict" and cache_hits != resets:
        raise RuntimeError(
            f"strict cache hits ({cache_hits}) do not match resets ({resets})"
        )


def get_binned_anchors(config: dict, scenario_file: pathlib.Path) -> np.ndarray:
    with np.load(scenario_file, allow_pickle=False) as data:
        dyn_traj = data["dyn_traj"]
        raw_candidates = compute_path_progress_candidate_steps(
            dyn_traj=dyn_traj,
            start_xy=data["init_pos"],
            goal_xy=data["goal"],
            arrival_steps=int(config.get("noise", {}).get("path_progress_arrival_steps", 230)),
            risk_distance_threshold=float(config.get("noise", {}).get("risk_distance_threshold", 10.0)),
            effective_horizon=min(int(config.get("env", {}).get("max_episode_steps", 512)), int(dyn_traj.shape[0])),
        )
    return bin_path_progress_candidate_steps(
        raw_candidates,
        stride=int(config.get("noise", {}).get("path_progress_candidate_stride", 1)),
    )


def build_precompute_env_kwargs(
    config: dict,
    cache_dir: pathlib.Path,
    cache_mode: str,
    scenario_dir: pathlib.Path | None = None,
) -> dict:
    env_kwargs = build_env_kwargs(config, scenario_dir or pathlib.Path("."))
    env_kwargs["load_on_reset"] = False
    env_kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    env_kwargs["kf_cache_dir"] = cache_dir
    env_kwargs["kf_cache_mode"] = cache_mode
    env_kwargs["filter_execution_mode"] = "precomputed"
    return env_kwargs


def precompute_binned_file(args: tuple) -> dict:
    config, scenario_file_text, cache_dir_text, cache_mode, forced_seed = args
    scenario_file = pathlib.Path(scenario_file_text)
    cache_dir = pathlib.Path(cache_dir_text)
    anchors = get_binned_anchors(config, scenario_file)
    env_kwargs = build_precompute_env_kwargs(config, cache_dir, cache_mode, scenario_file.parent)
    env = USVEnvMinimal(**env_kwargs)
    load_scenario_into_env(env, scenario_file)

    cache_hits = 0
    files_seen = 0
    unique_starts: set[int] = set()
    try:
        for anchor in anchors:
            env.forced_burst_start_step = int(anchor)
            env.reset(seed=int(forced_seed))
            cache_hits += int(bool(getattr(env, "_kf_cache_hit", False)))
            if env._kf_cache_path is not None and env._kf_cache_path.exists():
                files_seen += 1
            unique_starts.add(int(env.noise_injector._burst_start_step))
    finally:
        env.close()

    return {
        "file": str(scenario_file),
        "resets": int(len(anchors)),
        "binned_anchors": int(len(anchors)),
        "cache_hits": int(cache_hits),
        "cache_files_seen": int(files_seen),
        "unique_burst_starts": sorted(int(x) for x in unique_starts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/delay20_cov100_binned_s10.yaml")
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seeds", default="0:30", help="Range like 0:30 or comma list like 0,1,2")
    parser.add_argument("--cache-mode", choices=["write", "read_write", "read_strict"], default="read_write")
    parser.add_argument("--all-binned-candidates", action="store_true")
    parser.add_argument("--forced-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--audit-output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.audit_output is not None and args.audit_output.exists():
        raise FileExistsError(
            f"refusing to overwrite cache audit: {args.audit_output}"
        )

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    scenario_dir = pathlib.Path(args.scenario_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    files = sorted(scenario_dir.glob("*.npz"))
    files = files[max(0, int(args.start_index)): args.end_index]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .npz scenarios found under {scenario_dir}")
    seeds = parse_seed_spec(args.seeds)

    start_mode = str(config.get("noise", {}).get("start_mode", ""))
    if args.all_binned_candidates and start_mode != "path_progress_binned_candidate":
        raise ValueError("--all-binned-candidates requires noise.start_mode=path_progress_binned_candidate")

    t0 = time.perf_counter()
    cache_hits = 0
    files_seen = 0
    resets = 0
    unique_starts: set[int] = set()
    total_anchors = 0
    if args.all_binned_candidates:
        worker_args = [
            (config, str(scenario_file), str(cache_dir), args.cache_mode, int(args.forced_seed))
            for scenario_file in files
        ]
        workers = max(1, int(args.workers))
        if workers == 1:
            results = [precompute_binned_file(item) for item in worker_args]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(precompute_binned_file, item) for item in worker_args]
                for i, future in enumerate(as_completed(futures), start=1):
                    results.append(future.result())
                    if i % 25 == 0 or i == len(futures):
                        print(f"[CACHE] completed_files={i}/{len(futures)}", flush=True)

        for result in results:
            resets += int(result["resets"])
            total_anchors += int(result["binned_anchors"])
            cache_hits += int(result["cache_hits"])
            files_seen += int(result["cache_files_seen"])
            unique_starts.update(int(x) for x in result["unique_burst_starts"])
    else:
        env_kwargs = build_env_kwargs(config, scenario_dir)
        env_kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
        env_kwargs["kf_cache_dir"] = cache_dir
        env_kwargs["kf_cache_mode"] = args.cache_mode

        for seed in seeds:
            for scenario_file in files:
                env = USVEnvMinimal(**env_kwargs)
                select_single_scenario_file(env, scenario_file)
                env.reset(seed=seed)
                resets += 1
                cache_hits += int(bool(getattr(env, "_kf_cache_hit", False)))
                if env._kf_cache_path is not None and env._kf_cache_path.exists():
                    files_seen += 1
                unique_starts.add(int(env.noise_injector._burst_start_step))
                env.close()

    elapsed = time.perf_counter() - t0
    print(f"[CACHE] scenario_dir={scenario_dir}")
    print(f"[CACHE] cache_dir={cache_dir}")
    print(f"[CACHE] mode={args.cache_mode} all_binned_candidates={args.all_binned_candidates} workers={args.workers}")
    print(f"[CACHE] files={len(files)} seeds={len(seeds)} resets={resets}")
    if args.all_binned_candidates:
        print(f"[CACHE] binned_anchors={total_anchors}")
    print(f"[CACHE] cache_hits={cache_hits} cache_files_seen={files_seen} unique_burst_starts={len(unique_starts)}")
    print(f"[CACHE] elapsed_sec={elapsed:.3f}")
    if args.audit_output is not None:
        inventory = cache_inventory(cache_dir)
        validate_cache_inventory(
            inventory,
            files_seen=files_seen,
            resets=resets,
            cache_hits=cache_hits,
            cache_mode=args.cache_mode,
            all_binned_candidates=args.all_binned_candidates,
        )
        write_cache_audit(
            args.audit_output,
            {
                "status": "pass",
                "mode": args.cache_mode,
                "scenario_files": len(files),
                "resets": resets,
                "binned_anchors": total_anchors,
                "cache_hits": cache_hits,
                "cache_files_seen": files_seen,
                "unique_burst_starts": len(unique_starts),
                "elapsed_sec": elapsed,
                "cache_dir": str(cache_dir),
                **inventory,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
