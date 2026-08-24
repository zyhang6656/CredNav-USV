"""Generate or verify the final obs4/5/6 long-interaction map assets.

Run from the repository root:
  python scripts/generate_final_maps.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "simple_boat" / "map_generator" / "Generate_training_new.py"

TASKS = [
    {"n_obs": 4, "count": 1000, "output_dir": "simple_boat/assets/nav4_new_map", "prefix": "nav4_new", "seed": 20260604},
    {"n_obs": 4, "count": 200, "output_dir": "simple_boat/assets/eval4_new_map", "prefix": "eval4_new", "seed": 20261604},
    {"n_obs": 5, "count": 1000, "output_dir": "simple_boat/assets/nav5_new_map", "prefix": "nav5_new", "seed": 20260605},
    {"n_obs": 5, "count": 200, "output_dir": "simple_boat/assets/eval5_new_map", "prefix": "eval5_new", "seed": 20261605},
    {"n_obs": 6, "count": 1000, "output_dir": "simple_boat/assets/nav6_new_map", "prefix": "nav6_new", "seed": 20260606},
    {"n_obs": 6, "count": 200, "output_dir": "simple_boat/assets/eval6_new_map", "prefix": "eval6_new", "seed": 20261606},
]


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    for task in TASKS:
        output_dir = REPO_ROOT / task["output_dir"]
        existing = sorted(output_dir.glob("*.npz")) if output_dir.exists() else []
        if len(existing) == task["count"]:
            print(f"[SKIP] {task['output_dir']} already has {task['count']} maps")
            continue
        run(
            [
                sys.executable,
                str(GENERATOR),
                "--n-obs",
                str(task["n_obs"]),
                "--count",
                str(task["count"]),
                "--output-dir",
                task["output_dir"],
                "--prefix",
                task["prefix"],
                "--seed",
                str(task["seed"]),
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
