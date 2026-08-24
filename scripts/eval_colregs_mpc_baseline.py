from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
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
from scripts.train_obs3 import build_env_kwargs
from simple_boat.envs.colregs_mpc_baseline import COLREGSMPCBaseline
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def resolve_eval_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    sources = config.get("data", {}).get("eval_sources") or [
        {"label": f"obs{i}", "scenario_dir": f"simple_boat/assets/eval{i}_new_map"}
        for i in (3, 4, 5, 6)
    ]
    return [
        {
            "label": str(item["label"]),
            "scenario_dir": pathlib.Path(item["scenario_dir"]),
            "kf_cache_dir": item.get("kf_cache_dir"),
        }
        for item in sources
    ]


def env_config_for_source(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cache = dict(cfg.get("cache", {}))
    if source.get("kf_cache_dir"):
        cache["kf_cache_dir"] = source["kf_cache_dir"]
        cache.setdefault("kf_cache_mode", "read_strict")
    cfg["cache"] = cache
    return cfg


def make_env(config: dict[str, Any], source: dict[str, Any]) -> USVEnvMinimal:
    resolved = env_config_for_source(config, source)
    resolved["env"] = dict(resolved.get("env", {}))
    resolved["env"]["max_episode_steps"] = int(
        source.get("max_episode_steps", resolved["env"].get("max_episode_steps", 512))
    )
    kwargs = build_env_kwargs(resolved, source["scenario_dir"])
    kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    kwargs["load_on_reset"] = False
    kwargs["render_mode"] = False
    kwargs["render_freq"] = 100000
    return USVEnvMinimal(**kwargs)


def run_episode(
    *,
    controller: COLREGSMPCBaseline,
    config: dict[str, Any],
    source: dict[str, Any],
    map_path: pathlib.Path,
    seed: int,
) -> dict[str, Any]:
    env = make_env(config, source)
    load_scenario_into_env(env, map_path)
    _obs, _ = env.reset(seed=seed)
    max_steps = int(source.get("max_episode_steps", config.get("env", {}).get("max_episode_steps", 512)))
    done = False
    steps = 0
    ep_return = 0.0
    last_info: dict[str, Any] = {}
    points = [np.asarray(env.ship_state[:2], dtype=float).copy()]
    controller_times: list[float] = []
    step_rows: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    solver_failed_steps = 0

    while not done and steps < max_steps:
        prepared = env.prepared_control_latency_ns()
        t0 = time.perf_counter_ns()
        action, ctrl_info = controller.predict(env)
        decision_ns = time.perf_counter_ns() - t0
        solver_failed_steps += int(bool(ctrl_info.get("colregs_mpc_solver_failed", False)))
        _obs, reward, terminated, truncated, info = env.step(action)
        info.update(ctrl_info)
        latency = compose_control_latency_row(
            prepared=prepared,
            normalize_ns=0,
            policy_ns=decision_ns,
            safety_filter_ns=int(info.get("safety_filter_ns", 0)),
            actuator_mapping_ns=int(info.get("actuator_mapping_ns", 0)),
            cache_exact_match=bool(info.get("cache_exact_match", False)),
        )
        controller_times.append(float(latency["control_total_ms"]))
        role = str(ctrl_info.get("colregs_role", "NONE"))
        role_counts[role] = role_counts.get(role, 0) + 1
        ep_return += float(reward)
        done = bool(terminated or truncated)
        steps += 1
        step_rows.append(baseline_step_row(
            map_name=map_path.name,
            seed=seed,
            step=steps,
            env=env,
            info=info,
            fallback_action=action,
            latency=latency,
        ))
        last_info = dict(info)
        points.append(np.asarray(last_info.get("ship_xy", env.ship_state[:2]), dtype=float))

    reason = str(last_info.get("reason") or last_info.get("timeout_reason") or "unknown")
    min_actual = float(last_info.get("ep_min_actual_distance", last_info.get("min_actual_distance", np.inf)))
    row = {
        "map": map_path.name,
        "seed": int(seed),
        "steps": int(steps),
        "reason": reason,
        "return": float(ep_return),
        "path_length": float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))),
        "min_actual_distance": min_actual,
        "min_dcpa": float(last_info.get("ep_min_dcpa", last_info.get("min_dcpa", np.nan))),
        "colregs_compliance": float(last_info.get("colregs_ep_compliance", 1.0)),
        "colregs_mpc_ms_mean": float(np.mean(controller_times)) if controller_times else 0.0,
        "colregs_mpc_ms_p95": float(np.percentile(controller_times, 95)) if controller_times else 0.0,
        "control_latency_total_ms": float(sum(controller_times)),
        "control_latency_mean_ms": float(np.mean(controller_times)) if controller_times else 0.0,
        "colregs_mpc_gw_steps": int(role_counts.get("GW", 0)),
        "colregs_mpc_so_steps": int(role_counts.get("SO", 0)),
        "colregs_mpc_em_steps": int(role_counts.get("EM", 0)),
        "colregs_mpc_solver_failed_steps": int(solver_failed_steps),
        "colregs_mpc_solver_failed_rate": float(solver_failed_steps / max(1, steps)),
        "_step_rows": step_rows,
    }
    row.update(classify_episode_outcome(reason, min_actual, safety_distance=2.0))
    env.close()
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"episodes": len(rows)}
    for key in (
        "raw_success",
        "strict_success",
        "collision",
        "timeout",
        "unsafe_near_miss",
        "return",
        "path_length",
        "min_actual_distance",
        "min_dcpa",
        "colregs_compliance",
        "colregs_mpc_ms_mean",
        "colregs_mpc_ms_p95",
        "colregs_mpc_solver_failed_steps",
        "colregs_mpc_solver_failed_rate",
        "steps",
    ):
        vals = np.asarray([float(r[key]) for r in rows if np.isfinite(float(r[key]))], dtype=float)
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


def run_episode_batch_job(args: tuple[dict[str, Any], dict[str, Any], list[tuple[int, str, int]]]) -> list[tuple[int, dict[str, Any]]]:
    config, source, items = args
    controller = COLREGSMPCBaseline.from_config(config)
    source = dict(source)
    source["scenario_dir"] = pathlib.Path(source["scenario_dir"])
    return [
        (
            idx,
            run_episode(
                controller=controller,
                config=config,
                source=source,
                map_path=source["scenario_dir"] / map_name,
                seed=int(seed),
            ),
        )
        for idx, map_name, seed in items
    ]


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
    parser.add_argument("--config", default="configs/experiments/colregs_mpc_baseline.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    output_dir = pathlib.Path(args.output_dir or config.get("run", {}).get("output_root", "runs/colregs_mpc_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = COLREGSMPCBaseline.from_config(config)

    if args.protocol:
        protocol = load_evaluation_protocol(args.protocol, repo_root=REPO_ROOT)
        sources = protocol["sources"]
    else:
        sources = resolve_eval_sources(config)

    summaries: list[dict[str, Any]] = []
    wanted = set(args.labels or [])
    for source in sources:
        if wanted and source["label"] not in wanted:
            continue
        files = sorted(source["scenario_dir"].glob("*.npz"))
        if args.episodes is not None:
            files = files[: args.episodes]
        if not files:
            raise FileNotFoundError(f"No .npz maps found under {source['scenario_dir']}")
        if int(args.workers) > 1:
            workers = min(int(args.workers), len(files))
            chunks = [[] for _ in range(workers)]
            for idx, path in enumerate(files):
                chunks[idx % workers].append((idx, path.name, int(args.seed) + idx))
            jobs = [(config, {**source, "scenario_dir": str(source["scenario_dir"])}, chunk) for chunk in chunks if chunk]
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                indexed = [item for batch in pool.map(run_episode_batch_job, jobs) for item in batch]
            raw_rows = [row for _idx, row in sorted(indexed, key=lambda item: item[0])]
            step_rows = [step for row in raw_rows for step in row.pop("_step_rows")]
            rows = raw_rows
        else:
            rows = []
            step_rows = []
            progress_every = max(0, int(args.progress_every))
            episode_csv = output_dir / f"{source['label']}_episodes.csv"
            for idx, path in enumerate(files):
                row = run_episode(
                    controller=controller,
                    config=config,
                    source=source,
                    map_path=path,
                    seed=int(args.seed) + idx,
                )
                step_rows.extend(row.pop("_step_rows"))
                rows.append(row)
                if progress_every and ((idx + 1) % progress_every == 0 or idx + 1 == len(files)):
                    write_csv(episode_csv, rows)
                    write_csv(output_dir / f"{source['label']}_steps.csv", step_rows)
                    partial = aggregate(rows)
                    print(
                        f"[EVAL] {source['label']} {idx + 1}/{len(files)}: "
                        f"strict={partial.get('strict_success_mean', float('nan')):.3f} "
                        f"collision={partial.get('collision_mean', float('nan')):.3f} "
                        f"timeout={partial.get('timeout_mean', float('nan')):.3f} "
                        f"fail_rate={partial.get('colregs_mpc_solver_failed_rate_mean', float('nan')):.3f} "
                        f"last={row['reason']} steps={row['steps']}"
                    )
        write_csv(output_dir / f"{source['label']}_episodes.csv", rows)
        write_csv(output_dir / f"{source['label']}_steps.csv", step_rows)
        (output_dir / f"{source['label']}_resolved_config.json").write_text(
            json.dumps({
                "method": "colregs_mpcc",
                "env_kwargs": {
                    "dt": float(config.get("env", {}).get("dt", 0.1)),
                    "scenario_dir": str(source["scenario_dir"]),
                    "kf_cache_dir": str(source.get("kf_cache_dir", "")),
                    "max_episode_steps": int(source.get("max_episode_steps", 512)),
                },
            }, indent=2),
            encoding="utf-8",
        )
        summary = {"obs": source["label"], "scenario_dir": str(source["scenario_dir"])}
        summary.update(aggregate(rows))
        summaries.append(summary)
        print(
            f"[EVAL] {source['label']}: strict={summary.get('strict_success_mean', float('nan')):.3f} "
            f"collision={summary.get('collision_mean', float('nan')):.3f} "
            f"timeout={summary.get('timeout_mean', float('nan')):.3f}"
        )
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
