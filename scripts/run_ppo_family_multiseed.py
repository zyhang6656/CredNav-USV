"""Run the public PPO, Hetero-PPO, and CWVL five-seed training matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import yaml


METHODS = (
    (
        "ppo",
        "configs/experiments/mixed/"
        "mixed_obs3_6_delay20_cov100_binned_s10_5m_ppo_masked.yaml",
    ),
    (
        "hetero_ppo",
        "configs/experiments/mixed/"
        "mixed_obs3_6_delay20_cov100_binned_s10_5m_hetero_ppo.yaml",
    ),
    (
        "cwvl",
        "configs/experiments/mixed/"
        "mixed_obs3_6_delay20_cov100_binned_s10_5m_cwvl_critic_only.yaml",
    ),
)


@dataclass(frozen=True)
class Job:
    method: str
    seed: int
    base_config: Path
    resolved_config: Path
    run_dir: Path
    command: tuple[str, ...]


def _seeded_tag(tag: str, seed: int) -> str:
    if not tag.endswith("seed0"):
        raise ValueError(f"formal PPO-family run_tag must end in seed0: {tag}")
    return f"{tag[:-1]}{seed}"


def build_training_jobs(
    *, repo_root: Path, python_executable: Path, seeds: Iterable[int]
) -> list[Job]:
    root = Path(repo_root).resolve()
    python = str(python_executable)
    jobs: list[Job] = []
    for method, relative_config in METHODS:
        base_config = root / relative_config
        config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        run_root = root / config["run"]["run_root"]
        base_tag = str(config["run"]["run_tag"])
        for seed in seeds:
            seed = int(seed)
            run_tag = _seeded_tag(base_tag, seed)
            resolved_config = (
                root
                / "runs/ppo_family_trainseeds0_4"
                / "resolved_configs"
                / f"{method}_seed{seed}.yaml"
            )
            jobs.append(
                Job(
                    method=method,
                    seed=seed,
                    base_config=base_config,
                    resolved_config=resolved_config,
                    run_dir=run_root / run_tag,
                    command=(
                        python,
                        "-u",
                        str(root / "scripts/train_obs3.py"),
                        "--config",
                        str(resolved_config),
                        "--seed",
                        str(seed),
                    ),
                )
            )
    return jobs


def materialize_config(job: Job) -> None:
    config = yaml.safe_load(job.base_config.read_text(encoding="utf-8"))
    config["run"]["run_tag"] = job.run_dir.name
    job.resolved_config.parent.mkdir(parents=True, exist_ok=True)
    job.resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def training_complete(job: Job) -> bool:
    return all(
        (job.run_dir / relative).is_file()
        for relative in (
            "final_model.zip",
            "vec_normalize_final.pkl",
            "manifest.json",
            "resolved_config.yaml",
        )
    )


def _run_job(job: Job, *, repo_root: Path, log_root: Path) -> None:
    if training_complete(job):
        print(f"[SKIP] {job.method} seed{job.seed}", flush=True)
        return
    if job.run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite incomplete training directory: {job.run_dir}"
        )
    materialize_config(job)
    log_path = log_root / f"{job.method}_seed{job.seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            job.command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    if not training_complete(job):
        raise RuntimeError(f"incomplete training artifacts: {job.method} seed{job.seed}")


def run_training_jobs(
    *, repo_root: Path, jobs: list[Job], max_workers: int, log_root: Path
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_job, job, repo_root=repo_root, log_root=log_root)
            for job in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train PPO, Hetero-PPO, and CWVL for five independent seeds."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max-parallel-training", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)) or any(seed not in range(5) for seed in seeds):
        parser.error("--seeds must be unique values selected from 0, 1, 2, 3, 4")
    if args.max_parallel_training < 1:
        parser.error("--max-parallel-training must be positive")

    root = args.repo_root.resolve()
    jobs = build_training_jobs(
        repo_root=root,
        python_executable=Path(sys.executable),
        seeds=seeds,
    )
    if args.dry_run:
        print(f"training jobs: {len(jobs)}")
        for job in jobs:
            print(f"[{job.method}:seed{job.seed}] {' '.join(job.command)}")
        return 0

    run_training_jobs(
        repo_root=root,
        jobs=jobs,
        max_workers=args.max_parallel_training,
        log_root=root / "runs/ppo_family_trainseeds0_4/logs",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
