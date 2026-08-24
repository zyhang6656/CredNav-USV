from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

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
from scripts.training_io import _torch_load, resolve_sources
from scripts.train_lag_u_baseline import env_config_for_lag_u
from scripts.train_obs3 import build_env_kwargs
from simple_boat.envs.lag_u_baseline import (
    Actor,
    CostCritic,
    RewardCriticEnsemble,
    TrustFreeObservationWrapper,
    adaptive_cost_bound,
    ensemble_statistics,
    stable_standard_deviation,
    transition_cost,
    uncertainty_ratio,
)
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def make_env(
    config: dict[str, Any],
    source: dict[str, Any],
    *,
    online_exact: bool = False,
) -> TrustFreeObservationWrapper:
    resolved = env_config_for_lag_u(
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
    return TrustFreeObservationWrapper(USVEnvMinimal(**kwargs))


def load_model(
    model_path: pathlib.Path,
    device: torch.device,
) -> tuple[Actor, RewardCriticEnsemble, CostCritic, dict[str, Any]]:
    data = _torch_load(model_path, device)
    if int(data.get("model_version", 0)) != 2:
        raise ValueError("evaluation requires a linear-cost Lag-U model checkpoint v2")
    config = data["config"]
    cfg = config["lag_u"]
    hidden_dims = tuple(int(value) for value in cfg.get("hidden_dims", [256, 256]))
    actor = Actor(int(data["obs_dim"]), int(data["action_dim"]), hidden_dims).to(device)
    reward_critic = RewardCriticEnsemble(
        int(data["obs_dim"]),
        int(data["action_dim"]),
        int(cfg.get("ensemble_members", 3)),
        hidden_dims,
    ).to(device)
    cost_critic = CostCritic(int(data["obs_dim"]), int(data["action_dim"]), hidden_dims).to(device)
    actor.load_state_dict(data["actor"])
    reward_critic.load_state_dict(data["reward_critic"])
    cost_critic.load_state_dict(data["cost_critic"])
    actor.eval()
    reward_critic.eval()
    cost_critic.eval()
    return actor, reward_critic, cost_critic, config


@torch.no_grad()
def select_action(
    actor: Actor,
    reward_critic: RewardCriticEnsemble,
    cost_critic: CostCritic,
    obs: np.ndarray,
    *,
    device: torch.device,
    base_cost_bound: float,
    uncertainty_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    policy_action = actor(obs_tensor)
    member_values = reward_critic.policy_values(obs_tensor, policy_action)
    mean, variance = ensemble_statistics(member_values)
    standard_deviation = stable_standard_deviation(variance)
    ratio = uncertainty_ratio(mean, variance)
    bound = adaptive_cost_bound(
        ratio,
        base_bound=float(base_cost_bound),
        threshold=float(uncertainty_threshold),
    )
    cost_q = cost_critic(obs_tensor, policy_action)
    return policy_action.cpu().numpy()[0], {
        "q_mean": float(mean.reshape(-1)[0].item()),
        "q_std": float(standard_deviation.reshape(-1)[0].item()),
        "uncertainty_ratio": float(ratio.reshape(-1)[0].item()),
        "cost_q": float(cost_q.reshape(-1)[0].item()),
        "adaptive_cost_bound": float(bound.reshape(-1)[0].item()),
    }


def run_episode(
    *,
    actor: Actor,
    reward_critic: RewardCriticEnsemble,
    cost_critic: CostCritic,
    config: dict[str, Any],
    source: dict[str, Any],
    map_path: pathlib.Path,
    seed: int,
    device: torch.device,
    online_exact: bool = False,
) -> dict[str, Any]:
    env = make_env(config, source, online_exact=online_exact)
    try:
        load_scenario_into_env(env.env, map_path)
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        ep_return = 0.0
        ep_cost = 0.0
        info: dict[str, Any] = {}
        points = [np.asarray(env.env.ship_state[:2], dtype=float).copy()]
        q_means: list[float] = []
        q_stds: list[float] = []
        ratios: list[float] = []
        cost_qs: list[float] = []
        adaptive_bounds: list[float] = []
        control_times_ms: list[float] = []
        step_rows: list[dict[str, Any]] = []
        max_steps = int(source.get("max_episode_steps", config.get("env", {}).get("max_episode_steps", 512)))
        lag_u_cfg = config["lag_u"]

        while not done and steps < max_steps:
            prepared = env.env.prepared_control_latency_ns()
            decision_t0 = time.perf_counter_ns()
            action, diagnostics = select_action(
                actor,
                reward_critic,
                cost_critic,
                obs,
                device=device,
                base_cost_bound=float(lag_u_cfg.get("base_cost_bound", 0.1)),
                uncertainty_threshold=float(lag_u_cfg.get("uncertainty_threshold", 0.07)),
            )
            decision_ns = time.perf_counter_ns() - decision_t0
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
            ep_cost += transition_cost(info)
            q_means.append(diagnostics["q_mean"])
            q_stds.append(diagnostics["q_std"])
            ratios.append(diagnostics["uncertainty_ratio"])
            cost_qs.append(diagnostics["cost_q"])
            adaptive_bounds.append(diagnostics["adaptive_cost_bound"])
            done = bool(terminated or truncated)
            steps += 1
            step_rows.append(baseline_step_row(
                map_name=map_path.name,
                seed=seed,
                step=steps,
                env=env.env,
                info=info,
                fallback_action=action,
                latency=latency,
            ))
            points.append(np.asarray(info.get("ship_xy", env.env.ship_state[:2]), dtype=float))

        reason = str(info.get("reason", info.get("timeout_reason", "")))
        min_actual = float(info.get("ep_min_actual_distance", info.get("min_actual_distance", np.inf)))
        colregs_required = int(info.get("colregs_ep_required", 0))
        row = {
            "map": map_path.name,
            "seed": int(seed),
            "steps": int(steps),
            "reason": reason,
            "return": float(ep_return),
            "cost": float(ep_cost),
            "path_length": float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))),
            "min_actual_distance": min_actual,
            "colregs_compliance": (
                float(info.get("colregs_ep_compliance", np.nan)) if colregs_required else float("nan")
            ),
            "q_mean_episode": float(np.mean(q_means)) if q_means else float("nan"),
            "q_nonpositive_rate": float(np.mean(np.asarray(q_means) <= 0.0)) if q_means else float("nan"),
            "q_std_mean": float(np.mean(q_stds)) if q_stds else float("nan"),
            "q_std_max": float(np.max(q_stds)) if q_stds else float("nan"),
            "uncertainty_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
            "uncertainty_ratio_p95": float(np.percentile(ratios, 95)) if ratios else float("nan"),
            "uncertainty_ratio_max": float(np.max(ratios)) if ratios else float("nan"),
            "cost_q_initial": float(cost_qs[0]) if cost_qs else float("nan"),
            "cost_q_mean": float(np.mean(cost_qs)) if cost_qs else float("nan"),
            "cost_q_max": float(np.max(cost_qs)) if cost_qs else float("nan"),
            "cost_q_out_of_range_rate": (
                float(np.mean((np.asarray(cost_qs) < 0.0) | (np.asarray(cost_qs) > 1.0)))
                if cost_qs
                else float("nan")
            ),
            "adaptive_bound_mean": float(np.mean(adaptive_bounds)) if adaptive_bounds else float("nan"),
            "adaptive_bound_min": float(np.min(adaptive_bounds)) if adaptive_bounds else float("nan"),
            "control_latency_total_ms": float(sum(control_times_ms)),
            "control_latency_mean_ms": float(np.mean(control_times_ms)) if control_times_ms else 0.0,
            "_step_rows": step_rows,
        }
        row.update(classify_episode_outcome(reason, min_actual, safety_distance=2.0))
        return row
    finally:
        env.close()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    keys = (
        "raw_success",
        "strict_success",
        "collision",
        "timeout",
        "unsafe_near_miss",
        "return",
        "cost",
        "path_length",
        "min_actual_distance",
        "colregs_compliance",
        "steps",
        "q_mean_episode",
        "q_nonpositive_rate",
        "q_std_mean",
        "q_std_max",
        "uncertainty_ratio_mean",
        "uncertainty_ratio_p95",
        "uncertainty_ratio_max",
        "cost_q_initial",
        "cost_q_mean",
        "cost_q_max",
        "cost_q_out_of_range_rate",
        "adaptive_bound_mean",
        "adaptive_bound_min",
    )
    output: dict[str, float | int] = {"episodes": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        output[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        output[f"{key}_std"] = float(np.std(values)) if values.size else float("nan")
    return output


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--online-exact-timing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = pathlib.Path(args.model)
    actor, reward_critic, cost_critic, checkpoint_config = load_model(model_path, device)
    if args.config:
        config = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
        config["lag_u"] = checkpoint_config["lag_u"]
    else:
        config = checkpoint_config
    default_name = "eval_lag_u_usv"
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else model_path.parent / default_name
    if args.protocol:
        protocol = load_evaluation_protocol(args.protocol, repo_root=REPO_ROOT)
        sources = protocol["sources"]
    else:
        sources = resolve_sources(config, "eval_sources")

    summary_rows: list[dict[str, Any]] = []

    for source in sources:
        files = sorted(source["scenario_dir"].glob("*.npz"))[: int(args.episodes)]
        if len(files) < int(args.episodes):
            raise ValueError(f"{source['label']} has {len(files)} maps, requested {args.episodes}")
        raw_rows = [
            run_episode(
                actor=actor,
                reward_critic=reward_critic,
                cost_critic=cost_critic,
                config=config,
                source=source,
                map_path=path,
                seed=int(args.seed) + index,
                device=device,
                online_exact=bool(args.online_exact_timing),
            )
            for index, path in enumerate(files)
        ]
        step_rows = [step for row in raw_rows for step in row.pop("_step_rows")]
        rows = raw_rows
        write_csv(output_dir / f"{source['label']}_episodes.csv", rows)
        write_csv(output_dir / f"{source['label']}_steps.csv", step_rows)
        (output_dir / f"{source['label']}_resolved_config.json").write_text(
            json.dumps({
                "method": "lag_u",
                "config": (
                    str(pathlib.Path(args.config).resolve()) if args.config else None
                ),
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
        summary: dict[str, Any] = {"obs": source["label"]}
        summary.update(aggregate(rows))
        summary_rows.append(summary)
        print(
            f"[EVAL] {source['label']}: strict={summary['strict_success_mean']:.3f} "
            f"collision={summary['collision_mean']:.3f} "
            f"cost_q0={summary['cost_q_initial_mean']:.3f} ratio={summary['uncertainty_ratio_mean_mean']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
