"""Run the public DRL-VO and Lag-U five-seed training matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Job:
    phase: str
    method: str
    seed: int
    command: tuple[str, ...]


def build_training_jobs(
    *,
    repo_root: Path,
    python_executable: Path,
    seeds: Iterable[int],
) -> list[Job]:
    root = Path(repo_root)
    python = str(python_executable)
    jobs: list[Job] = []
    for method, script, config, max_steps in (
        (
            "drl_vo",
            "scripts/train_dqn_vo_baseline.py",
            "configs/experiments/dqn_vo_baseline.yaml",
            5_000_000,
        ),
        (
            "lag_u",
            "scripts/train_lag_u_baseline.py",
            "configs/experiments/lag_u_baseline.yaml",
            1_000_000,
        ),
    ):
        for seed in seeds:
            jobs.append(
                Job(
                    phase="train",
                    method=method,
                    seed=int(seed),
                    command=(
                        python,
                        "-u",
                        str(root / script),
                        "--config",
                        str(root / config),
                        "--seed",
                        str(seed),
                        "--max-timesteps",
                        str(max_steps),
                    ),
                )
            )
    return jobs


def method_run_dir(repo_root: Path, method: str, seed: int) -> Path:
    if method == "drl_vo":
        return repo_root / "runs/dqn_vo_baseline" / f"dqn_vo_uncertain_mixed_obs3_6_seed{seed}"
    if method == "lag_u":
        return (
            repo_root
            / "runs/lag_u_baseline"
            / f"lag_u_usv_absq_linear_cost_mixed_obs3_6_resumable_seed{seed}"
        )
    raise ValueError(f"unsupported method: {method}")



def training_command_for_run(job: Job, *, run_dir: Path) -> tuple[str, ...]:
    state_name = "trainer_state_latest.pkl" if job.method == "drl_vo" else "trainer_state_latest.pt"
    state_path = Path(run_dir) / state_name
    if state_path.is_file():
        return job.command + ("--resume", str(state_path.resolve()))
    return job.command


def training_complete(method: str, run_dir: Path) -> bool:
    if method == "drl_vo":
        return all(
            (run_dir / relative).is_file()
            for relative in (
                "model_5000000.zip",
                "vecnormalize_5000000.pkl",
                "checkpoints/checkpoint_5000000.zip",
                "checkpoints/vecnormalize_5000000.pkl",
                "manifest.json",
            )
        )
    if method == "lag_u":
        return all(
            (run_dir / relative).is_file()
            for relative in ("lag_u_500000.pt", "lag_u_1000000.pt", "manifest.json")
        )
    raise ValueError(f"unsupported method: {method}")



def _write_state(
    run_root: Path,
    *,
    phase: str,
    job: str,
    jobs: Iterable[str] | None = None,
) -> None:
    path = run_root / "state.json"
    temporary = run_root / "state.tmp.json"
    payload: dict[str, object] = {"phase": phase, "job": job}
    if jobs is not None:
        payload["jobs"] = list(jobs)
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_command(command: tuple[str, ...], *, repo_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _run_training_job(*, job: Job, repo_root: Path, run_root: Path) -> None:
    job_name = f"train_{job.method}_seed{job.seed}"
    run_dir = method_run_dir(repo_root, job.method, job.seed)
    _run_command(
        training_command_for_run(job, run_dir=run_dir),
        repo_root=repo_root,
        log_path=run_root / "logs" / f"{job_name}.log",
    )
    if not training_complete(job.method, run_dir):
        raise RuntimeError(f"training artifacts are incomplete after {job_name}")


def run_training_jobs(
    *,
    repo_root: Path,
    run_root: Path,
    training_jobs: list[Job],
    max_workers: int,
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    root = Path(repo_root).resolve()
    output = Path(run_root).resolve()
    remaining: list[Job] = []
    for job in training_jobs:
        job_name = f"train_{job.method}_seed{job.seed}"
        if training_complete(job.method, method_run_dir(root, job.method, job.seed)):
            print(f"[SKIP] complete {job_name}", flush=True)
        else:
            remaining.append(job)
    if not remaining:
        return

    for method, worker_limit in (("drl_vo", max_workers), ("lag_u", 1)):
        job_iterator = iter(job for job in remaining if job.method == method)
        active: dict[concurrent.futures.Future[None], Job] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as executor:
            while len(active) < worker_limit:
                try:
                    job = next(job_iterator)
                except StopIteration:
                    break
                active[executor.submit(_run_training_job, job=job, repo_root=root, run_root=output)] = job

            while active:
                active_names = [f"train_{job.method}_seed{job.seed}" for job in active.values()]
                _write_state(
                    output,
                    phase="training",
                    job=active_names[0],
                    jobs=active_names,
                )
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    active.pop(future)
                    future.result()
                    try:
                        job = next(job_iterator)
                    except StopIteration:
                        continue
                    active[executor.submit(
                        _run_training_job, job=job, repo_root=root, run_root=output
                    )] = job


def execute_workflow(
    *,
    repo_root: Path,
    run_root: Path,
    training_jobs: list[Job],
    max_parallel_training: int = 2,
) -> None:
    root = repo_root.resolve()
    output = run_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "runner.lock"
    try:
        lock = lock_path.open("x", encoding="ascii")
    except FileExistsError as exc:
        raise RuntimeError(f"workflow is already running: {lock_path}") from exc
    try:
        lock.write(str(__import__("os").getpid()))
        lock.close()
        run_training_jobs(
            repo_root=root,
            run_root=output,
            training_jobs=training_jobs,
            max_workers=max_parallel_training,
        )
        _write_state(output, phase="complete", job="complete")
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/drl_vo_lag_u_trainseeds0_4"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max-parallel-training", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)) or any(seed not in range(5) for seed in seeds):
        parser.error("--seeds must be unique values selected from 0, 1, 2, 3, 4")
    if args.max_parallel_training < 1:
        parser.error("--max-parallel-training must be positive")

    training = build_training_jobs(
        repo_root=args.repo_root.resolve(),
        python_executable=Path(sys.executable),
        seeds=seeds,
    )
    if args.dry_run:
        print(f"training jobs: {len(training)}")
        print(f"training workers: {args.max_parallel_training}")
        for job in training:
            print(f"[train:{job.method}:seed{job.seed}] {' '.join(job.command)}")
        print(f"run root: {args.run_root}")
        return 0
    execute_workflow(
        repo_root=args.repo_root,
        run_root=args.run_root,
        training_jobs=training,
        max_parallel_training=args.max_parallel_training,
    )
    print(f"[DONE] {args.run_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
