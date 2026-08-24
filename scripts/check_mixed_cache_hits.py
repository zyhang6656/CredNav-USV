"""Check strict KF cache hits for the mixed obs3-obs6 training setup."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pathlib
import sys
import time

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_helpers import load_scenario_into_env
from scripts.precompute_kf_cache import get_binned_anchors
from scripts.train_obs3 import build_env_kwargs
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


DEFAULT_CONFIG = "configs/experiments/mixed/mixed_obs3_6_delay20_cov100_binned_s10_5m_ppo_masked.yaml"


def resolve_datasets(config: dict) -> list[tuple[str, str, str | list[str]]]:
    data = config["data"]
    datasets: list[tuple[str, str, str | list[str]]] = []
    for source in data.get("train_sources", []):
        datasets.append(
            (
                f"train_{source['label']}",
                str(source["scenario_dir"]),
                str(source["kf_cache_dir"]),
            )
        )
    if not datasets:
        raise ValueError("mixed training config must declare train_sources")
    datasets.append(
        (
            "eval",
            str(data["eval_scenario_dir"]),
            config["cache"]["kf_cache_dir"],
        )
    )
    return datasets


def check_chunk(
    config_path: str, label: str, cache_dir: str | list[str], files: list[str]
) -> tuple[str, int, int]:
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    kwargs = build_env_kwargs(config, pathlib.Path("."))
    kwargs["scenario_dir"] = None
    kwargs["load_on_reset"] = False
    kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    kwargs["kf_cache_dir"] = cache_dir
    kwargs["kf_cache_mode"] = "read_strict"
    env = USVEnvMinimal(**kwargs)
    resets = 0
    try:
        for text in files:
            file_path = pathlib.Path(text)
            load_scenario_into_env(env, file_path)
            for anchor in get_binned_anchors(config, file_path):
                env.forced_burst_start_step = int(anchor)
                env.reset(seed=0)
                if not getattr(env, "_kf_cache_hit", False):
                    raise RuntimeError(f"cache not hit: {file_path} anchor={anchor}")
                resets += 1
    finally:
        env.close()
    return label, len(files), resets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    datasets = resolve_datasets(config)
    expected_files = {}
    tasks: list[tuple[str, str | list[str], list[str]]] = []
    for label, directory_text, cache_dir in datasets:
        files = [str(p) for p in sorted(pathlib.Path(directory_text).glob("*.npz"))]
        expected_files[label] = len(files)
        for i in range(0, len(files), int(args.chunk_size)):
            tasks.append((label, cache_dir, files[i:i + int(args.chunk_size)]))

    done_files = {label: 0 for label, _, _ in datasets}
    done_resets = {label: 0 for label, _, _ in datasets}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(check_chunk, args.config, label, cache_dir, files)
            for label, cache_dir, files in tasks
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            label, file_count, resets = future.result()
            done_files[label] += file_count
            done_resets[label] += resets
            print(
                f"[CACHE-CHECK] chunks={n}/{len(futures)} "
                + " ".join(
                    f"{name}={done_files[name]}/{expected_files[name]} "
                    f"resets={done_resets[name]}" for name in done_files
                ),
                flush=True,
            )

    print(f"[CACHE-CHECK] done elapsed_sec={time.perf_counter() - start:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
