from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import sys
import time
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import DQN

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_helpers import (
    baseline_step_row,
    classify_episode_outcome,
    load_evaluation_protocol,
    load_scenario_into_env,
)
from scripts.eval_transfer import compose_control_latency_row
from scripts.train_obs3 import build_env_kwargs
from scripts.train_dqn_vo_baseline import env_config_for_dqn_vo, resolve_sources
from simple_boat.envs.dqn_vo_baseline import DQNVOObservationRewardWrapper
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


class BoxVecNormalize:
    def __init__(self, path: pathlib.Path):
        with path.open("rb") as f:
            vn = pickle.load(f)
        self.mean = np.asarray(vn.obs_rms.mean, dtype=np.float32)
        self.var = np.asarray(vn.obs_rms.var, dtype=np.float32)
        self.clip_obs = float(getattr(vn, "clip_obs", 10.0))

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(obs, dtype=np.float32) - self.mean) / np.sqrt(self.var + 1e-8), -self.clip_obs, self.clip_obs)


def make_wrapped_env(
    config: dict[str, Any],
    source: dict[str, Any],
    *,
    online_exact: bool = False,
) -> DQNVOObservationRewardWrapper:
    resolved = env_config_for_dqn_vo(
        config, source["kf_cache_dir"], online_exact=online_exact
    )
    resolved["env"] = dict(resolved.get("env", {}))
    resolved["env"]["max_episode_steps"] = int(
        source.get("max_episode_steps", resolved["env"].get("max_episode_steps", 512))
    )
    kwargs = build_env_kwargs(resolved, source["scenario_dir"])
    kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    kwargs["load_on_reset"] = False
    kwargs["render_mode"] = False
    env = USVEnvMinimal(**kwargs)
    return DQNVOObservationRewardWrapper(
        env,
        fixed_surge=float(config["dqn_vo"]["fixed_surge"]),
        yaw_command=float(config["dqn_vo"]["yaw_command"]),
    )


def run_episode(
    *,
    model: DQN,
    vn: BoxVecNormalize | None,
    config: dict[str, Any],
    source: dict[str, Any],
    map_path: pathlib.Path,
    seed: int,
    online_exact: bool = False,
) -> dict[str, Any]:
    env = make_wrapped_env(config, source, online_exact=online_exact)
    load_scenario_into_env(env.env, map_path)
    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    steps = 0
    info: dict[str, Any] = {}
    points = [np.asarray(env.env.ship_state[:2], dtype=float).copy()]
    control_times_ms: list[float] = []
    step_rows: list[dict[str, Any]] = []
    max_steps = int(source.get("max_episode_steps", config.get("env", {}).get("max_episode_steps", 512)))

    while not done and steps < max_steps:
        prepared = env.env.prepared_control_latency_ns()
        decision_t0 = time.perf_counter_ns()
        obs_for_policy = vn.normalize_obs(obs) if vn is not None else obs
        action, _ = model.predict(obs_for_policy, deterministic=True)
        decision_ns = time.perf_counter_ns() - decision_t0
        continuous_action = env.map_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        latency = compose_control_latency_row(
            prepared=prepared,
            normalize_ns=0,
            policy_ns=decision_ns,
            safety_filter_ns=int(info.get("safety_filter_ns", 0)),
            actuator_mapping_ns=int(info.get("actuator_mapping_ns", 0)),
            cache_exact_match=bool(info.get("cache_exact_match", False)),
        )
        if online_exact and not latency["cache_exact_match"]:
            raise RuntimeError("online-exact timing requires an exact cache reference")
        control_times_ms.append(float(latency["control_total_ms"]))
        ep_return += float(reward)
        done = bool(terminated or truncated)
        steps += 1
        step_rows.append(baseline_step_row(
            map_name=map_path.name,
            seed=seed,
            step=steps,
            env=env.env,
            info=info,
            fallback_action=continuous_action,
            latency=latency,
        ))
        points.append(np.asarray(info.get("ship_xy", env.env.ship_state[:2]), dtype=float))

    reason = str(info.get("reason", info.get("timeout_reason", "")))
    min_actual = float(info.get("ep_min_actual_distance", info.get("min_actual_distance", np.inf)))
    outcome = classify_episode_outcome(reason, min_actual, safety_distance=2.0)
    path_length = 0.0
    for a, b in zip(points, points[1:]):
        path_length += float(np.linalg.norm(b - a))
    row = {
        "map": map_path.name,
        "seed": int(seed),
        "steps": int(steps),
        "reason": reason,
        "return": float(ep_return),
        "path_length": float(path_length),
        "min_actual_distance": min_actual,
        "colregs_compliance": float(info.get("colregs_ep_compliance", 1.0)),
        "control_latency_total_ms": float(sum(control_times_ms)),
        "control_latency_mean_ms": float(np.mean(control_times_ms)) if control_times_ms else 0.0,
        "_step_rows": step_rows,
    }
    row.update(outcome)
    env.close()
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics = [
        "raw_success",
        "strict_success",
        "collision",
        "timeout",
        "unsafe_near_miss",
        "return",
        "path_length",
        "min_actual_distance",
        "colregs_compliance",
        "steps",
    ]
    out: dict[str, float | int] = {"episodes": len(rows)}
    for key in metrics:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"{key}_std"] = float(np.std(vals)) if vals.size else float("nan")
    return out


def aggregate_control_latency(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    steps = sum(int(row["steps"]) for row in rows)
    total_ms = sum(float(row["control_latency_total_ms"]) for row in rows)
    return {
        "control_latency_steps": steps,
        "control_latency_mean_ms": total_ms / steps if steps else float("nan"),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dqn_vo_baseline.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--online-exact-timing", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    model_path = pathlib.Path(args.model)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else model_path.parent / "eval_dqn_vo"
    model = DQN.load(model_path)
    vn = BoxVecNormalize(pathlib.Path(args.vecnormalize)) if args.vecnormalize else None

    if args.protocol:
        protocol = load_evaluation_protocol(args.protocol, repo_root=REPO_ROOT)
        sources = protocol["sources"]
    else:
        sources = resolve_sources(config, "eval_sources")

    summary_rows: list[dict[str, Any]] = []
    for source in sources:
        label = source["label"]
        scenario_dir = source["scenario_dir"]
        files = sorted(scenario_dir.glob("*.npz"))[: int(args.episodes)]
        if len(files) < int(args.episodes):
            raise ValueError(f"{label} has {len(files)} maps, requested {args.episodes}")
        raw_rows = [
            run_episode(
                model=model, vn=vn, config=config, source=source,
                map_path=path, seed=int(args.seed) + i,
                online_exact=bool(args.online_exact_timing),
            )
            for i, path in enumerate(files)
        ]
        step_rows = [step for row in raw_rows for step in row.pop("_step_rows")]
        rows = raw_rows
        write_csv(output_dir / f"{label}_episodes.csv", rows)
        write_csv(output_dir / f"{label}_steps.csv", step_rows)
        (output_dir / f"{label}_resolved_config.json").write_text(
            json.dumps({
                "method": "drl_vo",
                "config": str(pathlib.Path(args.config).resolve()),
                "env_kwargs": {
                    "dt": float(config.get("env", {}).get("dt", 0.1)),
                    "scenario_dir": str(source["scenario_dir"]),
                    "kf_cache_dir": str(source["kf_cache_dir"]),
                    "max_episode_steps": int(source.get("max_episode_steps", 512)),
                    "filter_execution_mode": (
                        "online_exact" if args.online_exact_timing else "precomputed"
                    ),
                },
            }, indent=2),
            encoding="utf-8",
        )
        summary = {"obs": label}
        summary.update(aggregate(rows))
        summary.update(aggregate_control_latency(rows))
        summary_rows.append(summary)
        print(
            f"[EVAL] {label}: strict={summary['strict_success_mean']:.3f} "
            f"collision={summary['collision_mean']:.3f} timeout={summary['timeout_mean']:.3f}"
        )

    write_csv(output_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
