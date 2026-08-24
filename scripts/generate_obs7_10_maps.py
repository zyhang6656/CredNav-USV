"""Generate or verify the formal obs7--obs10 evaluation maps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MapGenerationJob:
    obstacles: int
    module: str
    output_dir: Path
    expected_count: int
    command: tuple[str, ...]


def build_generation_jobs(
    *, python_executable: Path = Path(sys.executable),
) -> list[MapGenerationJob]:
    """Return the ordered obs7--obs10 map-generation jobs."""
    jobs: list[MapGenerationJob] = []
    for obstacles, date in ((7, "20260726"), (8, "20260726"), (9, "20260727"), (10, "20260727")):
        module = (
            "simple_boat.map_generator."
            f"generate_evaluation_obs{obstacles}"
        )
        output_dir = (
            REPO_ROOT
            / "simple_boat"
            / "assets"
            / f"eval{obstacles}_paired_standard_t1024_{date}"
        )
        jobs.append(
            MapGenerationJob(
                obstacles=obstacles,
                module=module,
                output_dir=output_dir,
                expected_count=200,
                command=(str(python_executable), "-m", module),
            )
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered commands without running them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run generators even when all expected map files already exist.",
    )
    args = parser.parse_args()

    for job in build_generation_jobs():
        existing = list(job.output_dir.glob("*.npz")) if job.output_dir.is_dir() else []
        if len(existing) == job.expected_count and not args.force:
            print(
                f"[SKIP] obs{job.obstacles}: "
                f"{job.output_dir.relative_to(REPO_ROOT)} already contains "
                f"{job.expected_count} maps"
            )
            continue
        print("[RUN]", " ".join(job.command))
        if not args.dry_run:
            subprocess.run(job.command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
