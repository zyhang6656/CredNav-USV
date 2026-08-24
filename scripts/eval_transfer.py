"""Config-driven transfer evaluation for PPO/CW-VL checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import platform
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Optional

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_loading import load_unified_policy
from scripts.train_obs3 import build_env_kwargs
from scripts.eval_helpers import load_scenario_into_env, load_vn
import hashlib


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _code_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def build_eval_env_kwargs(
    config: dict[str, Any],
    scenario_dir: pathlib.Path,
    cache_mode: Optional[str],
    cache_dir: Optional[str],
    env_overrides: Optional[dict[str, Any]] = None,
    civo_overrides: Optional[dict[str, Any]] = None,
    colregs_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = dict(config)
    if env_overrides:
        env_cfg = dict(cfg.get("env", {}))
        env_cfg.update({k: v for k, v in env_overrides.items() if v is not None})
        cfg["env"] = env_cfg
    if cache_mode is not None or cache_dir is not None:
        cache_cfg = dict(cfg.get("cache", {}))
        if cache_mode is not None:
            cache_cfg["kf_cache_mode"] = cache_mode
        if cache_dir is not None:
            cache_cfg["kf_cache_dir"] = cache_dir
        cfg["cache"] = cache_cfg
    if civo_overrides:
        civo_cfg = dict(cfg.get("civo", {}))
        civo_cfg.update({k: v for k, v in civo_overrides.items() if v is not None})
        if bool(civo_cfg.get("shield_enabled", False)):
            civo_cfg["enabled"] = True
        cfg["civo"] = civo_cfg
    if colregs_overrides:
        colregs_cfg = dict(cfg.get("rc_colregs", cfg.get("colregs", {})))
        colregs_cfg.update({k: v for k, v in colregs_overrides.items() if v is not None})
        cfg["rc_colregs"] = colregs_cfg
    kwargs = build_env_kwargs(cfg, scenario_dir)
    kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    kwargs["load_on_reset"] = False
    kwargs["render_mode"] = False
    kwargs["render_freq"] = 100000
    return kwargs


def select_map_files(
    scenario_dir: pathlib.Path,
    *,
    map_start: int = 0,
    episodes: Optional[int] = None,
) -> list[pathlib.Path]:
    if map_start < 0 or (episodes is not None and episodes <= 0):
        raise ValueError("map_start must be nonnegative and episodes must be positive")
    files = sorted(pathlib.Path(scenario_dir).glob("*.npz"))
    selected = files[map_start:] if episodes is None else files[map_start : map_start + episodes]
    if not selected:
        raise FileNotFoundError(f"No selected .npz maps under {scenario_dir}")
    return selected


def estimated_topk_ids(env, k: int) -> list[int]:
    """Return obstacle IDs ordered by estimated range, then ID."""
    estimates = getattr(env, "obstacle_estimates", {})
    ranked = [
        (
            float(np.hypot(float(est.get("dx", 0.0)), float(est.get("dy", 0.0)))),
            int(obs_id),
        )
        for obs_id, est in estimates.items()
    ]
    return [obs_id for _, obs_id in sorted(ranked)[:max(0, int(k))]]


def build_policy_obs_estimated_topk(env, obs: dict[str, Any], k: int) -> dict[str, Any]:
    """Keep all physical targets, but feed the policy the k closest estimates."""
    k = int(k)
    fixed_slots = getattr(env, "POLICY_DYN_SLOTS", None)
    if fixed_slots is not None and k != int(fixed_slots):
        raise ValueError(
            f"requested K {k} does not match fixed policy slots {int(fixed_slots)}"
        )
    k = max(0, k)
    pos_scale = max(float(getattr(env, "obs_norm_range", 32.0)), 1e-6)
    v_scale = max(float(getattr(env, "u_max", 1.0)), 1e-6)
    dyn_rows: list[list[float]] = []
    dyn_mask: list[float] = []

    estimates = getattr(env, "obstacle_estimates", {})
    for obs_id in estimated_topk_ids(env, k):
        est = estimates[obs_id]
        x_loc = float(est.get("dx", 0.0))
        y_loc = float(est.get("dy", 0.0))
        vx_loc = float(est.get("vx", 0.0))
        vy_loc = float(est.get("vy", 0.0))
        rho_loc = float(np.hypot(x_loc, y_loc))
        dyn_rows.append([
            float(np.clip(x_loc / pos_scale, -1.0, 1.0)),
            float(np.clip(y_loc / pos_scale, -1.0, 1.0)),
            float(np.clip(vx_loc / v_scale, -1.0, 1.0)),
            float(np.clip(vy_loc / v_scale, -1.0, 1.0)),
            float(y_loc / rho_loc) if rho_loc > 1e-6 else 0.0,
            float(x_loc / rho_loc) if rho_loc > 1e-6 else 1.0,
            float(np.clip(float(est.get("trust", 1.0)), 0.0, 1.0)),
        ])
        dyn_mask.append(1.0)

    while len(dyn_rows) < k:
        dyn_rows.append([0.0] * 7)
        dyn_mask.append(0.0)
    return {
        "state": np.asarray(obs["state"], dtype=np.float32),
        "dyn": np.asarray(dyn_rows[:k], dtype=np.float32).reshape(k * 7),
        "dyn_mask": np.asarray(dyn_mask[:k], dtype=np.float32),
    }


def compose_control_latency_row(
    *,
    prepared: dict[str, int],
    normalize_ns: int,
    policy_ns: int,
    safety_filter_ns: int,
    actuator_mapping_ns: int,
    cache_exact_match: bool,
) -> dict[str, float | int | bool]:
    observation_ns = (
        prepared["coordinate_ns"]
        + prepared["observation_build_ns"]
        + normalize_ns
    )
    safety_ns = prepared["safety_precompute_ns"] + safety_filter_ns
    row = {
        "estimator_step": int(prepared["estimator_step"]),
        "measurement_sim_ms": prepared["measurement_sim_ns"] / 1e6,
        "filter_ms": prepared["filter_ns"] / 1e6,
        "trust_ms": prepared["trust_ns"] / 1e6,
        "observation_ms": observation_ns / 1e6,
        "policy_ms": policy_ns / 1e6,
        "safety_filter_ms": safety_ns / 1e6,
        "actuator_mapping_ms": actuator_mapping_ns / 1e6,
        "cache_exact_match": bool(cache_exact_match),
    }
    row["control_total_ms"] = float(sum(
        row[key]
        for key in (
            "filter_ms",
            "trust_ms",
            "observation_ms",
            "policy_ms",
            "safety_filter_ms",
            "actuator_mapping_ms",
        )
    ))
    return row


def build_step_diagnostic_row(
    *,
    map_name: str,
    seed: int,
    step: int,
    policy_action,
    info: dict[str, Any],
    env,
    policy_predict_latency_ms: float,
    step_wall_time_ms: float,
) -> dict[str, Any]:
    policy = list(
        info.get(
            "policy_action",
            np.asarray(policy_action).reshape(-1)[:2],
        )
    )
    executed = list(info.get("action", [np.nan, np.nan]))
    ship = np.asarray(
        getattr(env, "ship_state", np.full(6, np.nan)), dtype=float
    )
    mechanism = dict(getattr(env, "_last_civo_mechanism", {}))
    row = {
        "map": map_name,
        "seed": int(seed),
        "step": int(step),
        "controller_failure": bool(info.get("controller_failure", False)),
        "policy_surge": (
            float(policy[0]) if len(policy) > 0 else float("nan")
        ),
        "policy_yaw": (
            float(policy[1]) if len(policy) > 1 else float("nan")
        ),
        "action_surge": (
            float(executed[0]) if len(executed) > 0 else float("nan")
        ),
        "action_yaw": (
            float(executed[1]) if len(executed) > 1 else float("nan")
        ),
        "ship_x": float(info.get("ship_x", ship[0])),
        "ship_y": float(info.get("ship_y", ship[1])),
        "ship_psi": float(info.get("ship_psi", ship[2])),
        "ship_u": float(info.get("ship_u", ship[3])),
        "ship_v": float(info.get("ship_v", ship[4])),
        "ship_r": float(info.get("ship_r", ship[5])),
        "n_port_cmd": float(info.get("n_port_cmd", np.nan)),
        "n_stbd_cmd": float(info.get("n_stbd_cmd", np.nan)),
        "tau_u_cmd_raw": float(info.get("tau_u_cmd_raw", np.nan)),
        "tau_r_cmd_raw": float(info.get("tau_r_cmd_raw", np.nan)),
        "tau_u_rate_limit_active": bool(
            info.get("tau_u_rate_limit_active", False)
        ),
        "tau_u_cmd": float(info.get("tau_u_cmd", np.nan)),
        "dot_tau_u_cmd": float(info.get("dot_tau_u_cmd", np.nan)),
        "tau_r_cmd": float(info.get("tau_r_cmd", np.nan)),
        "civo_shield_changed": bool(
            info.get(
                "civo_shield_changed",
                getattr(env, "_last_civo_shield_changed", False),
            )
        ),
        "civo_shield_delta": float(
            info.get(
                "civo_shield_delta",
                getattr(env, "_last_civo_shield_delta", 0.0),
            )
        ),
        "civo_qp_success": bool(
            info.get(
                "civo_qp_success",
                getattr(env, "_last_civo_qp_success", False),
            )
        ),
        "civo_qp_solver": str(
            info.get(
                "civo_qp_solver",
                getattr(env, "_last_civo_qp_solver", ""),
            )
        ),
        "civo_qp_status": str(
            info.get(
                "civo_qp_status",
                getattr(env, "_last_civo_qp_status", ""),
            )
        ),
        "civo_qp_iter": int(
            info.get(
                "civo_qp_iter",
                getattr(env, "_last_civo_qp_iter", 0),
            )
        ),
        "civo_constraint_count": int(
            info.get(
                "civo_constraint_count",
                getattr(env, "_last_civo_constraint_count", 0),
            )
        ),
        "civo_obstacle_id": int(
            info.get(
                "civo_obstacle_id",
                getattr(env, "_last_civo_obstacle_id", -1),
            )
        ),
        "civo_cbf_h": float(
            info.get(
                "civo_cbf_h",
                getattr(env, "_last_civo_h", np.nan),
            )
        ),
        "civo_cbf_residual": float(
            info.get(
                "civo_cbf_residual",
                getattr(env, "_last_civo_residual", np.nan),
            )
        ),
        "civo_qp_slack": float(
            info.get(
                "civo_qp_slack",
                getattr(env, "_last_civo_slack", 0.0),
            )
        ),
        "credible_geometry_fallback_count": int(
            info.get("credible_geometry_fallback_count", 0)
        ),
        "credible_geometry_fallback_ids": ";".join(
            str(value)
            for value in info.get("credible_geometry_fallback_ids", ())
        ),
        "credible_geometry_fallback_reasons": " | ".join(
            str(value)
            for value in info.get(
                "credible_geometry_fallback_reasons", ()
            )
        ),
        "civo_inside_count": int(info.get("civo_inside_count", 0)),
        "civo_clearance_min": float(
            info.get("civo_clearance_min", np.nan)
        ),
        "min_actual_distance": float(
            info.get("min_actual_distance", np.nan)
        ),
        "min_dcpa": float(info.get("min_dcpa", np.nan)),
        "colregs_active_count": int(
            info.get("colregs_active_count", 0)
        ),
        "colregs_required_count": int(
            info.get("colregs_required_count", 0)
        ),
        "colregs_compliance": float(
            info.get("colregs_compliance", 1.0)
        ),
        "civo_shield_latency_ms": float(
            info.get(
                "civo_shield_latency_ms",
                getattr(env, "_last_civo_shield_latency_ms", 0.0),
            )
        ),
        "policy_predict_latency_ms": float(
            policy_predict_latency_ms
        ),
        "step_wall_time_ms": float(step_wall_time_ms),
    }
    for key, default in {
        "vo_cbf_active_vo_count": 0,
        "vo_cbf_active_hard_count": 0,
        "vo_cbf_active_vo_row_count": 0,
        "vo_cbf_active_hard_row_count": 0,
        "vo_cbf_slack_sum": 0.0,
        "vo_cbf_slack_max": 0.0,
        "vo_cbf_vo_residual_min": float("nan"),
        "vo_cbf_hard_residual_min": float("nan"),
        "vo_cbf_delta_tau_u": 0.0,
        "vo_cbf_delta_tau_r": 0.0,
        "vo_cbf_delta_thrust_common": 0.0,
        "vo_cbf_delta_thrust_differential": 0.0,
        "vo_cbf_safety_row_rank": 0,
        "vo_cbf_fallback": 0,
        "vo_cbf_fallback_reason": "",
        "vo_cbf_domain_exit_count": 0,
        "vo_cbf_domain_exit_ids": "",
        "vo_cbf_yaw_sensitivity_max": 0.0,
        "vo_cbf_predictive_residual_error_max": float("nan"),
        "vo_cbf_predictive_rejection_count": 0,
        "vo_cbf_predictive_domain_exit_count": 0,
        "cbf_vo_prediction_steps": 0,
        "vo_cbf_failure_reason": "",
    }.items():
        row[key] = info.get(key, mechanism.get(key, default))
    return row


def run_one_episode(
    *,
    model,
    vn,
    map_path: pathlib.Path,
    env_kwargs: dict[str, Any],
    seed: int,
    deterministic: bool,
    max_steps: int,
    policy_top_k_estimated: Optional[int] = None,
    freeze_cache_tail: bool = False,
) -> dict[str, Any]:
    from simple_boat.envs.usv_env_minimal import (
        SafetyFilterRuntimeError,
        USVEnvMinimal,
    )

    env = USVEnvMinimal(**env_kwargs)
    try:
        load_scenario_into_env(env, map_path)
        obs, _ = env.reset(seed=seed)
        scenario_T = int(getattr(env, "scenario_T", 0))
        if freeze_cache_tail:
            reset_max_steps = int(env.max_episode_steps)
            if getattr(env, "filter_execution_mode", "") != "online_exact":
                raise ValueError("freeze-cache-tail requires online_exact")
            if not bool(getattr(env, "_kf_cache_hit", False)):
                raise ValueError(
                    "freeze-cache-tail requires a strict cache hit"
                )
            if scenario_T <= 1 or int(max_steps) <= reset_max_steps:
                raise ValueError(
                    "runtime horizon must exceed the reset horizon"
                )
            env.max_episode_steps = int(max_steps)
        done = False
        steps = 0
        ep_return = 0.0
        last_info: dict[str, Any] = {}
        path_points = [np.asarray(env.ship_state[:2], dtype=float).copy()]
        step_wall_times: list[float] = []
        latency_rows: list[dict[str, Any]] = []
        controller_failure_message = ""
        policy_k = max(
            0,
            int(
                policy_top_k_estimated
                if policy_top_k_estimated is not None
                else getattr(env, "POLICY_DYN_SLOTS", 6)
            ),
        )
        previous_policy_membership: Optional[frozenset[int]] = None
        policy_top_k_switches = 0
        policy_top_k_seen: set[int] = set()

        while not done and steps < max_steps:
            frozen_tail = bool(
                freeze_cache_tail and steps >= scenario_T - 1
            )
            if frozen_tail and env.filter_execution_mode != "precomputed":
                env.filter_execution_mode = "precomputed"
            t0 = time.perf_counter()
            prepared = env.prepared_control_latency_ns()
            policy_ids = tuple(env._last_policy_obstacle_ids)
            if len(policy_ids) > policy_k:
                raise RuntimeError("environment returned too many policy obstacles")
            normalize_started = time.perf_counter_ns()
            obs_for_policy = vn.normalize_obs(obs) if vn is not None else obs
            normalize_ns = time.perf_counter_ns() - normalize_started
            policy_started = time.perf_counter_ns()
            action, _ = model.predict(
                obs_for_policy, deterministic=deterministic
            )
            policy_ns = time.perf_counter_ns() - policy_started
            policy_membership = frozenset(policy_ids)
            if (
                previous_policy_membership is not None
                and policy_membership != previous_policy_membership
            ):
                policy_top_k_switches += 1
            previous_policy_membership = policy_membership
            policy_top_k_seen.update(policy_ids)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except SafetyFilterRuntimeError as exc:
                step_wall_time = time.perf_counter() - t0
                step_wall_times.append(step_wall_time)
                steps += 1
                controller_failure_message = str(exc)
                last_info = {
                    "reason": "controller_failure",
                    "controller_failure": True,
                }
                latency_row = build_step_diagnostic_row(
                    map_name=map_path.name,
                    seed=seed,
                    step=steps,
                    policy_action=action,
                    info=last_info,
                    env=env,
                    policy_predict_latency_ms=policy_ns / 1e6,
                    step_wall_time_ms=step_wall_time * 1000.0,
                )
                latency_row.update(compose_control_latency_row(
                    prepared=prepared,
                    normalize_ns=normalize_ns,
                    policy_ns=policy_ns,
                    safety_filter_ns=int(getattr(
                        env,
                        "_last_civo_shield_latency_ns",
                        1e6 * float(getattr(
                            env, "_last_civo_shield_latency_ms", 0.0
                        )),
                    )),
                    actuator_mapping_ns=0,
                    cache_exact_match=bool(
                        getattr(env, "filter_execution_mode", "")
                        == "online_exact"
                        and getattr(env, "_online_cache_mismatches", 0) == 0
                    ),
                ))
                if freeze_cache_tail:
                    latency_row.update({
                        "frozen_tail": frozen_tail,
                        "cache_exact_required": not frozen_tail,
                        "scenario_step": int(env.dyn_step),
                        "scenario_T": scenario_T,
                    })
                latency_rows.append(latency_row)
                break

            step_wall_time = time.perf_counter() - t0
            step_wall_times.append(step_wall_time)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            steps += 1
            last_info = dict(info)
            path_points.append(
                np.asarray(
                    last_info.get("ship_xy", env.ship_state[:2]),
                    dtype=float,
                )
            )
            latency_row = build_step_diagnostic_row(
                map_name=map_path.name,
                seed=seed,
                step=steps,
                policy_action=action,
                info=last_info,
                env=env,
                policy_predict_latency_ms=policy_ns / 1e6,
                step_wall_time_ms=step_wall_time * 1000.0,
            )
            latency_row.update(compose_control_latency_row(
                prepared=prepared,
                normalize_ns=normalize_ns,
                policy_ns=policy_ns,
                safety_filter_ns=int(last_info["safety_filter_ns"]),
                actuator_mapping_ns=int(
                    last_info["actuator_mapping_ns"]
                ),
                cache_exact_match=bool(
                    last_info.get("cache_exact_match", False)
                ),
            ))
            if freeze_cache_tail:
                latency_row.update({
                    "frozen_tail": frozen_tail,
                    "cache_exact_required": not frozen_tail,
                    "scenario_step": int(env.dyn_step),
                    "scenario_T": scenario_T,
                })
            latency_rows.append(latency_row)

    finally:
        env.close()

    ep_min_actual = float(
        last_info.get(
            "ep_min_actual_distance",
            getattr(env, "_ep_min_actual_dist", np.nan),
        )
    )
    ep_min_dcpa = float(
        last_info.get(
            "ep_min_dcpa",
            getattr(env, "_ep_min_dcpa", np.nan),
        )
    )
    reason = str(
        last_info.get("reason")
        or last_info.get("timeout_reason")
        or "unknown"
    )
    reason_key = reason.strip().lower()
    success_reasons = {"goal_reached"}
    collision_reasons = {
        "collision", "static_obs", "dynamic_obs", "out_of_bounds"
    }
    timeout_reasons = {"max_steps", "scenario_end", "timeout"}
    failure_reasons = {"controller_failure"}
    if reason_key not in (
        success_reasons | collision_reasons | timeout_reasons | failure_reasons
    ):
        raise RuntimeError(
            f"unexpected terminal reason {reason!r} for {map_path.name}"
        )
    success = int(reason_key in success_reasons)
    collision = int(reason_key in collision_reasons)
    timeout = int(reason_key in timeout_reasons)
    controller_failure = int(reason_key in failure_reasons)
    assert success + collision + timeout + controller_failure == 1
    strict_success = int(success and not collision and ep_min_actual >= 2.0)
    unsafe_near_miss = int(not collision and ep_min_actual < 2.0)
    path_length = 0.0
    if len(path_points) > 1:
        path_length = float(
            sum(
                np.linalg.norm(
                    path_points[i] - path_points[i - 1]
                )
                for i in range(1, len(path_points))
            )
        )
    direct_distance = float(np.linalg.norm(
        np.asarray(env.goal, dtype=float)
        - np.asarray(env.initial_position, dtype=float)
    ))
    path_efficiency = (
        float(direct_distance / path_length)
        if success and path_length > 1e-9
        else float("nan")
    )

    summary = {
        "map": map_path.name,
        "seed": int(seed),
        "steps": int(steps),
        "reason": reason,
        "success": success,
        "strict_success": strict_success,
        "goal_reached": success,
        "collision": collision,
        "timeout": timeout,
        "controller_failure": controller_failure,
        "controller_failure_message": controller_failure_message,
        "fallback_count": int(
            getattr(env, "_vo_cbf_ep_fallback_count", 0)
        ),
        "vo_domain_exit_count": int(
            getattr(env, "_vo_cbf_ep_domain_exit_count", 0)
        ),
        "unsafe_near_miss": unsafe_near_miss,
        "return": float(ep_return),
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "min_actual_distance": ep_min_actual,
        "min_dcpa": ep_min_dcpa,
        "mean_trust": float(last_info.get("mean_trust", np.nan)),
        "burst_start_step": int(getattr(env.noise_injector, "_burst_start_step", -1)),
        "burst_mode": str(getattr(env.noise_injector, "_burst_mode_used", "none")),
        "kf_cache_hit": bool(getattr(env, "_kf_cache_hit", False)),
        "kf_cache_path": str(getattr(env, "_kf_cache_path", "") or ""),
        "civo_active_ratio": float(last_info.get("civo_ep_active_ratio", 0.0)),
        "civo_shield_ratio": float(last_info.get("civo_ep_shield_ratio", 0.0)),
        "civo_delta_mean": float(last_info.get("civo_ep_delta_mean", 0.0)),
        "civo_delta_max": float(last_info.get("civo_ep_delta_max", 0.0)),
        "civo_slack_mean": float(last_info.get("civo_ep_slack_mean", 0.0)),
        "civo_slack_max": float(last_info.get("civo_ep_slack_max", 0.0)),
        "civo_latency_mean_ms": float(last_info.get("civo_ep_latency_mean_ms", 0.0)),
        "civo_latency_p95_ms": float(last_info.get("civo_ep_latency_p95_ms", 0.0)),
        "civo_latency_p99_ms": float(last_info.get("civo_ep_latency_p99_ms", 0.0)),
        "civo_latency_max_ms": float(last_info.get("civo_ep_latency_max_ms", 0.0)),
        "credible_geometry_fallback_steps": int(
            last_info.get("credible_geometry_fallback_ep_steps", 0)
        ),
        "credible_geometry_fallback_obstacle_steps": int(
            last_info.get("credible_geometry_fallback_ep_obstacle_steps", 0)
        ),
        "colregs_required": int(last_info.get("colregs_ep_required", 0)),
        "colregs_compliant": int(last_info.get("colregs_ep_compliant", 0)),
        "colregs_active": int(last_info.get("colregs_ep_active", 0)),
        "colregs_compliance": float(last_info.get("colregs_ep_compliance", 1.0)),
        "policy_top_k_switches": int(policy_top_k_switches),
        "policy_top_k_unique_obstacles": int(len(policy_top_k_seen)),
        "physical_obstacle_count": int(env.dyn_obs_num),
        "policy_obstacle_slots": int(env.POLICY_DYN_SLOTS),
        "online_cache_checks": int(
            getattr(env, "_online_cache_checks", 0)
        ),
        "online_cache_mismatches": int(
            getattr(env, "_online_cache_mismatches", 0)
        ),
        "step_wall_time_mean_ms": float(np.mean(step_wall_times) * 1000.0) if step_wall_times else 0.0,
        "step_wall_time_p95_ms": float(np.percentile(step_wall_times, 95) * 1000.0) if step_wall_times else 0.0,
        "_latency_rows": latency_rows,
    }
    def finite_step_values(key: str) -> np.ndarray:
        values = np.asarray(
            [float(row.get(key, np.nan)) for row in latency_rows],
            dtype=float,
        )
        return values[np.isfinite(values)]

    vo_residuals = finite_step_values("vo_cbf_vo_residual_min")
    hard_residuals = finite_step_values("vo_cbf_hard_residual_min")
    delta_tau_u = finite_step_values("vo_cbf_delta_tau_u")
    delta_tau_r = finite_step_values("vo_cbf_delta_tau_r")
    predictive_errors = finite_step_values(
        "vo_cbf_predictive_residual_error_max"
    )
    summary.update(
        {
            "vo_cbf_active_vo_steps": int(
                sum(
                    int(row.get("vo_cbf_active_vo_count", 0)) > 0
                    for row in latency_rows
                )
            ),
            "vo_cbf_active_hard_steps": int(
                sum(
                    int(row.get("vo_cbf_active_hard_count", 0)) > 0
                    for row in latency_rows
                )
            ),
            "vo_cbf_active_vo_row_count_max": int(max(
                (
                    int(row.get("vo_cbf_active_vo_row_count", 0))
                    for row in latency_rows
                ),
                default=0,
            )),
            "vo_cbf_active_hard_row_count_max": int(max(
                (
                    int(row.get("vo_cbf_active_hard_row_count", 0))
                    for row in latency_rows
                ),
                default=0,
            )),
            "vo_cbf_slack_sum": float(
                sum(
                    float(row.get("vo_cbf_slack_sum", 0.0))
                    for row in latency_rows
                )
            ),
            "vo_cbf_slack_max": float(
                max(
                    (
                        float(row.get("vo_cbf_slack_max", 0.0))
                        for row in latency_rows
                    ),
                    default=0.0,
                )
            ),
            "vo_cbf_vo_residual_min": (
                float(np.min(vo_residuals))
                if vo_residuals.size
                else float("nan")
            ),
            "vo_cbf_hard_residual_min": (
                float(np.min(hard_residuals))
                if hard_residuals.size
                else float("nan")
            ),
            "vo_cbf_delta_tau_u_mean": (
                float(np.mean(delta_tau_u))
                if delta_tau_u.size
                else 0.0
            ),
            "vo_cbf_delta_tau_u_max_abs": (
                float(np.max(np.abs(delta_tau_u)))
                if delta_tau_u.size
                else 0.0
            ),
            "vo_cbf_delta_tau_r_mean": (
                float(np.mean(delta_tau_r))
                if delta_tau_r.size
                else 0.0
            ),
            "vo_cbf_delta_tau_r_max_abs": (
                float(np.max(np.abs(delta_tau_r)))
                if delta_tau_r.size
                else 0.0
            ),
            "vo_cbf_safety_row_rank_max": int(
                max(
                    (
                        int(row.get("vo_cbf_safety_row_rank", 0))
                        for row in latency_rows
                    ),
                    default=0,
                )
            ),
            "vo_cbf_yaw_sensitivity_max": float(
                max(
                    (
                        float(row.get("vo_cbf_yaw_sensitivity_max", 0.0))
                        for row in latency_rows
                    ),
                    default=0.0,
                )
            ),
            "vo_cbf_predictive_residual_error_max": (
                float(np.max(predictive_errors))
                if predictive_errors.size
                else float("nan")
            ),
            "vo_cbf_predictive_rejection_count": int(
                sum(
                    int(row.get("vo_cbf_predictive_rejection_count", 0))
                    for row in latency_rows
                )
            ),
            "vo_cbf_predictive_domain_exit_count": int(
                sum(
                    int(row.get("vo_cbf_predictive_domain_exit_count", 0))
                    for row in latency_rows
                )
            ),
            "cbf_vo_prediction_steps": int(
                max(
                    (
                        int(row.get("cbf_vo_prediction_steps", 0))
                        for row in latency_rows
                    ),
                    default=0,
                )
            ),
        }
    )
    if freeze_cache_tail:
        required_rows = [
            row for row in latency_rows if row["cache_exact_required"]
        ]
        summary.update({
            "runtime_max_steps": int(max_steps),
            "freeze_start_step": int(scenario_T),
            "frozen_tail_steps": int(sum(
                bool(row["frozen_tail"]) for row in latency_rows
            )),
            "cache_verified_until_step": int(
                required_rows[-1]["step"] if required_rows else 0
            ),
            "final_goal_distance": float(np.linalg.norm(
                np.asarray(env.ship_state[:2], dtype=float)
                - np.asarray(env.goal, dtype=float)
            )),
        })
    return summary


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "success",
        "strict_success",
        "goal_reached",
        "collision",
        "timeout",
        "controller_failure",
        "fallback_count",
        "vo_domain_exit_count",
        "vo_cbf_predictive_rejection_count",
        "vo_cbf_predictive_domain_exit_count",
        "cbf_vo_prediction_steps",
        "vo_cbf_predictive_residual_error_max",
        "unsafe_near_miss",
        "return",
        "path_length",
        "min_actual_distance",
        "min_dcpa",
        "steps",
        "kf_cache_hit",
        "civo_active_ratio",
        "civo_shield_ratio",
        "civo_delta_mean",
        "civo_delta_max",
        "civo_slack_mean",
        "civo_slack_max",
        "civo_latency_mean_ms",
        "civo_latency_p95_ms",
        "civo_latency_p99_ms",
        "civo_latency_max_ms",
        "credible_geometry_fallback_steps",
        "credible_geometry_fallback_obstacle_steps",
        "colregs_required",
        "colregs_compliant",
        "colregs_active",
        "colregs_compliance",
        "policy_top_k_switches",
        "policy_top_k_unique_obstacles",
        "step_wall_time_mean_ms",
        "step_wall_time_p95_ms",
    ]
    summary: dict[str, Any] = {"episodes": len(rows)}
    for key in metrics:
        values = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=float)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        if key in {"min_actual_distance", "min_dcpa"}:
            summary[f"{key}_p5"] = float(np.percentile(values, 5))
    for key in (
        "success",
        "strict_success",
        "collision",
        "timeout",
        "controller_failure",
    ):
        summary[f"{key}_count"] = int(
            sum(int(row[key]) for row in rows)
        )
    summary["fallback_count_total"] = int(
        sum(int(row["fallback_count"]) for row in rows)
    )
    summary["episodes_with_fallback"] = int(
        sum(int(row["fallback_count"]) > 0 for row in rows)
    )
    summary["vo_domain_exit_count_total"] = int(
        sum(int(row["vo_domain_exit_count"]) for row in rows)
    )
    successful = [
        row for row in rows if int(row["success"]) == 1
    ]
    if successful:
        summary["successful_steps_mean"] = float(
            np.mean([float(row["steps"]) for row in successful])
        )
        summary["successful_path_length_mean"] = float(
            np.mean(
                [float(row["path_length"]) for row in successful]
            )
        )
        summary["successful_travel_time_mean_s"] = float(
            0.1 * summary["successful_steps_mean"]
        )
        summary["successful_speed_mean_mps"] = float(
            np.mean(
                [
                    float(row["path_length"])
                    / max(0.1 * float(row["steps"]), 1e-9)
                    for row in successful
                ]
            )
        )
    return summary


def aggregate_step_latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    civo = np.asarray([float(row["civo_shield_latency_ms"]) for row in rows], dtype=float)
    policy = np.asarray([float(row["policy_predict_latency_ms"]) for row in rows], dtype=float)
    wall = np.asarray([float(row["step_wall_time_ms"]) for row in rows], dtype=float)
    status_counts = Counter(str(row.get("civo_qp_status", "")) for row in rows if str(row.get("civo_qp_status", "")))
    solver_counts = Counter(str(row.get("civo_qp_solver", "")) for row in rows if str(row.get("civo_qp_solver", "")))
    constraint_counts = [int(row.get("civo_constraint_count", 0)) for row in rows]
    slacks = np.asarray([float(row.get("civo_qp_slack", 0.0)) for row in rows], dtype=float)
    action_deltas = np.asarray(
        [float(row.get("civo_shield_delta", 0.0)) for row in rows], dtype=float
    )
    residuals = np.asarray(
        [float(row.get("civo_cbf_residual", np.nan)) for row in rows],
        dtype=float,
    )
    finite_residuals = residuals[np.isfinite(residuals)]
    fallback_counts = np.asarray(
        [int(row.get("credible_geometry_fallback_count", 0)) for row in rows],
        dtype=int,
    )
    vo_slack_sums = np.asarray(
        [float(row.get("vo_cbf_slack_sum", 0.0)) for row in rows],
        dtype=float,
    )
    vo_slack_maxima = np.asarray(
        [float(row.get("vo_cbf_slack_max", 0.0)) for row in rows],
        dtype=float,
    )
    vo_ranks = np.asarray(
        [int(row.get("vo_cbf_safety_row_rank", 0)) for row in rows],
        dtype=int,
    )
    vo_residuals = np.asarray(
        [
            float(row.get("vo_cbf_vo_residual_min", np.nan))
            for row in rows
        ],
        dtype=float,
    )
    hard_residuals = np.asarray(
        [
            float(row.get("vo_cbf_hard_residual_min", np.nan))
            for row in rows
        ],
        dtype=float,
    )
    finite_vo_residuals = vo_residuals[np.isfinite(vo_residuals)]
    finite_hard_residuals = hard_residuals[
        np.isfinite(hard_residuals)
    ]
    predictive_errors = np.asarray(
        [
            float(row.get("vo_cbf_predictive_residual_error_max", np.nan))
            for row in rows
        ],
        dtype=float,
    )
    predictive_errors = predictive_errors[np.isfinite(predictive_errors)]

    def truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    summary = {
        "step_latency_count": int(civo.size),
        "civo_step_latency_mean_ms": float(np.mean(civo)),
        "civo_step_latency_p95_ms": float(np.percentile(civo, 95)),
        "civo_step_latency_p99_ms": float(np.percentile(civo, 99)),
        "civo_step_latency_max_ms": float(np.max(civo)),
        "civo_step_slack_mean": float(np.mean(slacks)),
        "civo_step_slack_max": float(np.max(slacks)),
        "civo_step_action_delta_mean": float(np.mean(action_deltas)),
        "civo_step_action_delta_max": float(np.max(action_deltas)),
        "policy_predict_latency_mean_ms": float(np.mean(policy)),
        "policy_predict_latency_p95_ms": float(np.percentile(policy, 95)),
        "policy_predict_latency_p99_ms": float(np.percentile(policy, 99)),
        "policy_predict_latency_max_ms": float(np.max(policy)),
        "step_wall_time_mean_ms": float(np.mean(wall)),
        "step_wall_time_p95_ms": float(np.percentile(wall, 95)),
        "step_wall_time_p99_ms": float(np.percentile(wall, 99)),
        "step_wall_time_max_ms": float(np.max(wall)),
        "civo_qp_status_counts": dict(status_counts),
        "civo_qp_solver_counts": dict(solver_counts),
        "civo_qp_success_count": int(sum(truthy(row.get("civo_qp_success", False)) for row in rows)),
        "civo_shield_changed_count": int(sum(truthy(row.get("civo_shield_changed", False)) for row in rows)),
        "vo_cbf_fallback_step_count": int(
            sum(truthy(row.get("vo_cbf_fallback", False)) for row in rows)
        ),
        "vo_cbf_domain_exit_obstacle_step_count": int(
            sum(int(row.get("vo_cbf_domain_exit_count", 0)) for row in rows)
        ),
        "civo_constraint_count": int(sum(constraint_counts)),
        "credible_geometry_fallback_step_count": int(np.count_nonzero(fallback_counts)),
        "credible_geometry_fallback_obstacle_step_count": int(np.sum(fallback_counts)),
        "civo_cbf_residual_min": (
            float(np.min(finite_residuals)) if finite_residuals.size else None
        ),
        "vo_cbf_slack_sum": float(np.sum(vo_slack_sums)),
        "vo_cbf_slack_max": float(np.max(vo_slack_maxima)),
        "vo_cbf_safety_row_rank_max": int(np.max(vo_ranks)),
        "vo_cbf_yaw_sensitivity_max": float(
            max(
                float(row.get("vo_cbf_yaw_sensitivity_max", 0.0))
                for row in rows
            )
        ),
        "vo_cbf_vo_residual_min": (
            float(np.min(finite_vo_residuals))
            if finite_vo_residuals.size
            else None
        ),
        "vo_cbf_hard_residual_min": (
            float(np.min(finite_hard_residuals))
            if finite_hard_residuals.size
            else None
        ),
        "vo_cbf_predictive_residual_error_max": (
            float(np.max(predictive_errors))
            if predictive_errors.size
            else None
        ),
        "vo_cbf_predictive_rejection_count": int(
            sum(
                int(row.get("vo_cbf_predictive_rejection_count", 0))
                for row in rows
            )
        ),
        "vo_cbf_predictive_domain_exit_count": int(
            sum(
                int(row.get("vo_cbf_predictive_domain_exit_count", 0))
                for row in rows
            )
        ),
        "cbf_vo_prediction_steps": int(
            max(
                (int(row.get("cbf_vo_prediction_steps", 0)) for row in rows),
                default=0,
            )
        ),
        "vo_cbf_active_vo_row_count_max": int(max(
            (int(row.get("vo_cbf_active_vo_row_count", 0)) for row in rows),
            default=0,
        )),
        "vo_cbf_active_hard_row_count_max": int(max(
            (int(row.get("vo_cbf_active_hard_row_count", 0)) for row in rows),
            default=0,
        )),
    }
    online_fields = (
        "measurement_sim_ms",
        "filter_ms",
        "trust_ms",
        "observation_ms",
        "policy_ms",
        "safety_filter_ms",
        "actuator_mapping_ms",
        "control_total_ms",
    )
    online_rows = [all(key in row for key in online_fields) for row in rows]
    if any(online_rows) and not all(online_rows):
        raise ValueError("online latency fields must be present on every row")
    if all(online_rows):
        component_fields = online_fields[1:-1]
        for row in rows:
            components = [float(row[key]) for key in component_fields]
            total = float(row["control_total_ms"])
            values = [float(row[key]) for key in online_fields]
            if any(not np.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(
                    "online latency values must be finite and non-negative"
                )
            if abs(total - sum(components)) > 1e-9:
                raise ValueError(
                    "control_total_ms does not equal its components"
                )
        for field in online_fields:
            values = np.asarray([float(row[field]) for row in rows])
            summary.update({
                f"{field}_mean": float(np.mean(values)),
                f"{field}_p50": float(np.percentile(values, 50)),
                f"{field}_p95": float(np.percentile(values, 95)),
                f"{field}_p99": float(np.percentile(values, 99)),
                f"{field}_max": float(np.max(values)),
            })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vec-normalize", required=True)
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--map-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-mode", choices=["off", "read", "write", "read_write", "read_strict"], default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--filter-execution-mode",
        choices=("precomputed", "online_exact"),
        default=None,
    )
    parser.add_argument("--actuator-tau-u-dot-max", type=float, default=None)
    parser.add_argument("--actuator-n-dot-max", type=float, default=None)
    parser.add_argument("--civo-enable", action="store_true")
    parser.add_argument("--civo-shield", action="store_true")
    parser.add_argument("--civo-confidence", type=float, default=None)
    parser.add_argument("--civo-shield-distance", type=float, default=None)
    parser.add_argument("--civo-shield-tcpa-horizon", type=float, default=None)
    parser.add_argument(
        "--civo-shield-method",
        choices=(
            "corecbf",
            "relaxed_vo_cbf",
            "cbf_vo",
        ),
        default=None,
    )
    parser.add_argument("--vo-cbf-alpha-vo", type=float, default=None)
    parser.add_argument("--vo-cbf-alpha-c", type=float, default=None)
    parser.add_argument("--vo-cbf-k-u", type=float, default=None)
    parser.add_argument("--vo-cbf-k-vo", type=float, default=None)
    parser.add_argument("--corecbf-surge-accel", type=float, default=None)
    parser.add_argument("--corecbf-turn-accel", type=float, default=None)
    parser.add_argument("--corecbf-turn-direction", type=int, choices=(-1, 1), default=None)
    parser.add_argument(
        "--corecbf-variant",
        choices=("deterministic", "credibility_colregs"),
        default=None,
    )
    parser.add_argument(
        "--civo-shield-gate-mode",
        choices=("native", "all_obstacles", "distance_tcpa"),
        default=None,
    )
    parser.add_argument("--corecbf-gain", type=float, default=None)
    parser.add_argument("--corecbf-safety-distance", type=float, default=None)
    parser.add_argument("--corecbf-colregs-reference-scale", type=float, default=None)
    parser.add_argument("--corecbf-tau-u-weight", type=float, default=None)
    parser.add_argument("--corecbf-tau-r-weight", type=float, default=None)
    parser.add_argument(
        "--corecbf-separable-slack", action="store_true", default=None
    )
    parser.add_argument(
        "--corecbf-prediction-steps", type=int, choices=(0,), default=0
    )
    parser.add_argument("--corecbf-slack-weight", type=float, default=None)
    parser.add_argument("--corecbf-osqp-max-iter", type=int, default=None)
    parser.add_argument("--corecbf-osqp-eps-abs", type=float, default=None)
    parser.add_argument("--corecbf-osqp-eps-rel", type=float, default=None)
    parser.add_argument("--corecbf-osqp-polishing", dest="corecbf_osqp_polishing", action="store_true", default=None)
    parser.add_argument("--corecbf-osqp-no-polishing", dest="corecbf_osqp_polishing", action="store_false")
    parser.add_argument("--rc-colregs-enable", action="store_true")
    parser.add_argument("--rc-colregs-reward-weight", type=float, default=None)
    parser.add_argument("--rc-colregs-d-safe", type=float, default=None)
    parser.add_argument("--rc-colregs-tau", type=float, default=None)
    parser.add_argument("--rc-colregs-kappa", type=float, default=None)
    parser.add_argument("--min-strict-success", type=float, default=None)
    parser.add_argument("--ground-truth-obstacles", action="store_true")
    parser.add_argument(
        "--policy-top-k-estimated",
        "--policy-top-k-actual",
        dest="policy_top_k_estimated",
        type=int,
        choices=(6,),
        default=None,
        help="Feed the policy the K closest estimated obstacles while the environment keeps all physical obstacles",
    )
    parser.add_argument("--runtime-max-steps", type=int, default=None)
    parser.add_argument("--freeze-cache-tail", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    reset_max_steps = int(
        config.get("env", {}).get("max_episode_steps", 512)
    )
    if args.freeze_cache_tail:
        if (
            args.runtime_max_steps is None
            or int(args.runtime_max_steps) <= reset_max_steps
        ):
            parser.error(
                "--freeze-cache-tail requires --runtime-max-steps greater "
                "than the configured horizon"
            )
        if args.filter_execution_mode != "online_exact":
            parser.error(
                "--freeze-cache-tail requires --filter-execution-mode "
                "online_exact"
            )
        if args.cache_mode != "read_strict":
            parser.error(
                "--freeze-cache-tail requires --cache-mode read_strict"
            )
    elif args.runtime_max_steps is not None:
        parser.error(
            "--runtime-max-steps requires --freeze-cache-tail"
        )
    scenario_dir = pathlib.Path(args.scenario_dir)
    map_files = select_map_files(
        scenario_dir,
        map_start=args.map_start,
        episodes=args.episodes,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "resolved": output_dir / f"{args.label}_resolved_config.json",
        "episodes": output_dir / f"{args.label}_episodes.csv",
        "steps": output_dir / f"{args.label}_steps.csv",
        "summary": output_dir / f"{args.label}_summary.json",
    }
    existing = [path for path in artifact_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    civo_overrides = {
        "enabled": True if args.civo_enable else None,
        "shield_enabled": True if args.civo_shield else None,
        "confidence": args.civo_confidence,
        "shield_distance": args.civo_shield_distance,
        "shield_tcpa_horizon": args.civo_shield_tcpa_horizon,
        "shield_method": args.civo_shield_method,
        "shield_gate_mode": args.civo_shield_gate_mode,
        "vo_cbf_alpha_vo": args.vo_cbf_alpha_vo,
        "vo_cbf_alpha_c": args.vo_cbf_alpha_c,
        "vo_cbf_k_u": args.vo_cbf_k_u,
        "vo_cbf_k_vo": args.vo_cbf_k_vo,
        "surge_accel": args.corecbf_surge_accel,
        "turn_accel": args.corecbf_turn_accel,
        "turn_direction": args.corecbf_turn_direction,
        "corecbf_variant": args.corecbf_variant,
        "cbf_gain": args.corecbf_gain,
        "safety_distance": args.corecbf_safety_distance,
        "colregs_reference_scale": args.corecbf_colregs_reference_scale,
        "qp_tau_u_weight": args.corecbf_tau_u_weight,
        "qp_tau_r_weight": args.corecbf_tau_r_weight,
        "separable_slack_enabled": args.corecbf_separable_slack,
        "shared_slack_weight": args.corecbf_slack_weight,
        "osqp_max_iter": args.corecbf_osqp_max_iter,
        "osqp_eps_abs": args.corecbf_osqp_eps_abs,
        "osqp_eps_rel": args.corecbf_osqp_eps_rel,
        "osqp_polishing": args.corecbf_osqp_polishing,
    }
    colregs_overrides = {
        "enabled": True if args.rc_colregs_enable else None,
        "reward_weight": args.rc_colregs_reward_weight,
        "d_safe": args.rc_colregs_d_safe,
        "tau": args.rc_colregs_tau,
        "kappa": args.rc_colregs_kappa,
    }
    env_overrides = {
        "actuator_tau_u_dot_max": args.actuator_tau_u_dot_max,
        "actuator_n_dot_max": args.actuator_n_dot_max,
        "filter_execution_mode": args.filter_execution_mode,
        "use_filter": False if args.ground_truth_obstacles else None,
    }
    env_kwargs = build_eval_env_kwargs(
        config,
        scenario_dir,
        args.cache_mode,
        args.cache_dir,
        env_overrides=env_overrides,
        civo_overrides=civo_overrides,
        colregs_overrides=colregs_overrides,
    )
    paper_values = {
        "vo_cbf_alpha_vo": 10.0,
        "vo_cbf_alpha_c": 10.0,
        "vo_cbf_k_u": (
            1.0
            if env_kwargs["civo_shield_method"] == "relaxed_vo_cbf"
            else 2.0
        ),
        "vo_cbf_k_vo": (
            1000.0
            if env_kwargs["civo_shield_method"] == "relaxed_vo_cbf"
            else 50.0
        ),
    }
    if "baseline_provenance" in config:
        for key, value in paper_values.items():
            if float(env_kwargs[key]) != value:
                raise ValueError(
                    f"baseline provenance requires {key}={value}, "
                    f"got {env_kwargs[key]}"
                )
        prediction_steps = config["baseline_provenance"].get(
            "prediction_steps"
        )
        if prediction_steps is not None and (
            type(prediction_steps) is not int or prediction_steps != 1
        ):
            raise ValueError(
                "CBF-VO provenance must record prediction_steps=1"
            )
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    vec_normalize_path = pathlib.Path(args.vec_normalize).resolve()
    resolved = {
        "label": args.label,
        "config": str(pathlib.Path(args.config).resolve()),
        "scenario_dir": str(scenario_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "vec_normalize": str(vec_normalize_path),
        "vec_normalize_sha256": _sha256(vec_normalize_path),
        "code_commit": _code_commit(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "env_kwargs": env_kwargs,
        "cbf_vo_prediction_steps": 1,
        "corecbf_prediction_steps": int(args.corecbf_prediction_steps),
        "cache_mode": args.cache_mode,
        "cache_dir": args.cache_dir,
        "map_start": int(args.map_start),
        "seed_rule": "seed = base_seed + map_start + selected_index",
        "baseline_provenance": dict(
            config.get("baseline_provenance", {})
        ),
        "maps": [
            {
                "path": str(path.resolve()),
                "name": path.name,
                "seed": int(args.seed + args.map_start + index),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(map_files)
        ],
    }
    if args.freeze_cache_tail:
        resolved["runtime_protocol"] = {
            "reset_max_episode_steps": reset_max_steps,
            "runtime_max_steps": int(args.runtime_max_steps),
            "freeze_cache_tail": True,
            "formal_latency_excludes_frozen_tail": True,
        }
    artifact_paths["resolved"].write_text(
        json.dumps(
            resolved,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    model = load_unified_policy(args.checkpoint, device=args.device)
    vn = load_vn(args.vec_normalize, missing_ok=False)
    max_steps = int(args.runtime_max_steps or reset_max_steps)

    rows = []
    latency_rows: list[dict[str, Any]] = []
    for idx, map_path in enumerate(map_files):
        global_index = args.map_start + idx
        try:
            row = run_one_episode(
                model=model,
                vn=vn,
                map_path=map_path,
                env_kwargs=env_kwargs,
                seed=args.seed + global_index,
                deterministic=args.deterministic,
                max_steps=max_steps,
                policy_top_k_estimated=args.policy_top_k_estimated,
                freeze_cache_tail=args.freeze_cache_tail,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Evaluation failed on episode {idx + 1}/{len(map_files)}, "
                f"map={map_path.name}, seed={args.seed + global_index}: {exc}"
            ) from exc
        latency_rows.extend(row.pop("_latency_rows", []))
        rows.append(row)
        if (idx + 1) % 50 == 0 or idx + 1 == len(map_files):
            recent = rows[-50:]
            print(
                f"[EVAL] {args.label} {idx + 1}/{len(map_files)} "
                f"strict={np.mean([r['strict_success'] for r in recent]):.3f} "
                f"coll={np.mean([r['collision'] for r in recent]):.3f}"
            )

    with artifact_paths["episodes"].open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if latency_rows:
        with artifact_paths["steps"].open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(latency_rows[0].keys()))
            writer.writeheader()
            writer.writerows(latency_rows)

    summary = aggregate(rows)
    summary.update(aggregate_step_latencies(latency_rows))
    summary.update(
        {
            "label": args.label,
            "checkpoint": args.checkpoint,
            "vec_normalize": args.vec_normalize,
            "scenario_dir": str(scenario_dir),
            "config": args.config,
            "cache_mode": args.cache_mode,
            "cache_dir": args.cache_dir,
            "map_start": args.map_start,
            "policy_top_k_estimated": args.policy_top_k_estimated,
            "baseline_provenance": dict(config.get("baseline_provenance", {})),
            "effective_vo_cbf_parameters": {
                **{key: float(env_kwargs[key]) for key in paper_values},
                "cbf_vo_prediction_steps": 1,
            },
            "ground_truth_obstacles": bool(args.ground_truth_obstacles),
            "env_overrides": {k: v for k, v in env_overrides.items() if v is not None},
            "civo_overrides": {k: v for k, v in civo_overrides.items() if v is not None},
            "colregs_overrides": {k: v for k, v in colregs_overrides.items() if v is not None},
        },
    )
    artifact_paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[DONE] {args.label}: strict={summary.get('strict_success_mean', float('nan')):.3f} "
        f"collision={summary.get('collision_mean', float('nan')):.3f} "
        f"timeout={summary.get('timeout_mean', float('nan')):.3f}"
    )
    if args.min_strict_success is not None:
        strict = float(summary.get("strict_success_mean", float("nan")))
        if not np.isfinite(strict) or strict < float(args.min_strict_success):
            print(f"[GATE_FAIL] strict_success_mean={strict:.3f} < {args.min_strict_success:.3f}")
            return 2
        print(f"[GATE_PASS] strict_success_mean={strict:.3f} >= {args.min_strict_success:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
