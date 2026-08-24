"""Minimal USV environment for CW-VL vs PPO experiment.

STRIPPED:
  - CI-VO and RC-COLREGs are optional and disabled by default for checkpoint compatibility.

OBSERVATION (identical for PPO and CW-VL):
  {"state": (13,), "dyn": (42,), "dyn_mask": (6,)}
  - t_min is in info ONLY, never in observation.
  - Trainer injects trust into rollout buffer after collection.
  - Trust used ONLY in CW-VL critic loss.
"""

import hashlib
import json
import os
import pathlib
import time
from typing import Optional, Dict
import gymnasium
import numpy as np
from gymnasium import spaces
import math

from simple_boat.envs.dynamics import prop_thrust, update_usv_full_model
from simple_boat.envs.corecbf_lie import (
    CredibleGeometryDomainError,
    aggregate_colregs_duty,
    colregs_duty,
    colregs_reference_control,
    corecbf_terms,
    credibility_aware_corecbf_terms,
    credible_covariance,
    credible_geometry_scale,
    solve_corecbf_qp,
)
from simple_boat.envs.relaxed_vo_cbf import (
    RelaxedVOCBFDomainError,
    RelaxedVOCBFNoVerifiedAction,
    hard_collision_barrier_value,
    hard_collision_cbf_terms,
    otter_acceleration_maps,
    relaxed_vo_cbf_terms,
    solve_relaxed_affine_qp,
    solve_relaxed_vo_cbf_qp,
    vo_barrier_value,
)
from simple_boat.utils.utils import (
    normalize_angle_0_to_2pi, distance_to_goal,
    in_bounds, world_to_local,
    draw_usv, draw_obstacles,
)
from simple_boat.envs.filter_trust import InnovationConsistencyTrust, OracleTrust
from simple_boat.envs.noise_models import BurstMismatchInjector


class DynamicObstacleCapacityError(ValueError):
    pass


class OnlineFilterCacheMismatch(RuntimeError):
    pass


VO_CBF_FINITE_DIFFERENCE_FRACTION = 1.0e-3
CBF_VO_PREDICTION_STEPS = 1


class SafetyFilterRuntimeError(RuntimeError):
    """A safety filter failed before its command was physically executed."""


def _empty_vo_cbf_diagnostics() -> dict[str, float | int | str]:
    return {
        "vo_cbf_active_vo_count": 0,
        "vo_cbf_active_hard_count": 0,
        "vo_cbf_active_vo_row_count": 0,
        "vo_cbf_active_hard_row_count": 0,
        "vo_cbf_domain_exit_count": 0,
        "vo_cbf_domain_exit_ids": "",
        "vo_cbf_fallback": 0,
        "vo_cbf_fallback_reason": "",
        "vo_cbf_slack_sum": 0.0,
        "vo_cbf_slack_max": 0.0,
        "vo_cbf_vo_residual_min": float("nan"),
        "vo_cbf_hard_residual_min": float("nan"),
        "vo_cbf_delta_tau_u": 0.0,
        "vo_cbf_delta_tau_r": 0.0,
        "vo_cbf_delta_thrust_common": 0.0,
        "vo_cbf_delta_thrust_differential": 0.0,
        "vo_cbf_safety_row_rank": 0,
        "vo_cbf_yaw_sensitivity_max": 0.0,
        "vo_cbf_predictive_residual_error_max": float("nan"),
        "vo_cbf_predictive_rejection_count": 0,
        "vo_cbf_predictive_domain_exit_count": 0,
        "cbf_vo_prediction_steps": 0,
        "vo_cbf_failure_reason": "",
    }


def compute_path_progress_candidate_steps(
    dyn_traj: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    arrival_steps: int,
    risk_distance_threshold: float,
    effective_horizon: int,
) -> list[int]:
    """Return proxy path-risk steps where any obstacle is near ownship path."""
    if dyn_traj is None or dyn_traj.shape[1] <= 0:
        return []
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    denom = max(1.0, float(arrival_steps))
    threshold = float(risk_distance_threshold)
    T = int(dyn_traj.shape[0])
    scan_end = min(T, max(1, int(effective_horizon)))
    n = int(dyn_traj.shape[1])
    candidates: list[int] = []
    for k in range(1, scan_end):
        alpha = float(np.clip(float(k) / denom, 0.0, 1.0))
        ship_xy = start + alpha * (goal - start)
        obs_xy = dyn_traj[k, :n, :2].astype(float)
        if bool(np.any(np.linalg.norm(obs_xy - ship_xy[None, :], axis=1) < threshold)):
            candidates.append(int(k))
    return candidates


def compute_path_progress_min_distances(
    dyn_traj: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    arrival_steps: int,
    effective_horizon: int,
) -> np.ndarray:
    """Return min proxy path distance to any obstacle at each step."""
    if dyn_traj is None or dyn_traj.shape[1] <= 0:
        return np.asarray([], dtype=float)
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    denom = max(1.0, float(arrival_steps))
    T = int(dyn_traj.shape[0])
    scan_end = min(T, max(1, int(effective_horizon)))
    n = int(dyn_traj.shape[1])
    distances = np.full(scan_end, np.inf, dtype=float)
    for k in range(1, scan_end):
        alpha = float(np.clip(float(k) / denom, 0.0, 1.0))
        ship_xy = start + alpha * (goal - start)
        obs_xy = dyn_traj[k, :n, :2].astype(float)
        distances[k] = float(np.min(np.linalg.norm(obs_xy - ship_xy[None, :], axis=1)))
    return distances


def bin_path_progress_candidate_steps(candidates: list[int], stride: int) -> np.ndarray:
    """Compress adjacent candidate steps into median anchors."""
    if not candidates:
        return np.asarray([], dtype=np.int32)
    stride = max(1, int(stride))
    if stride == 1:
        return np.asarray(sorted(set(int(k) for k in candidates)), dtype=np.int32)

    unique = sorted(set(int(k) for k in candidates))
    anchors: list[int] = []
    run: list[int] = []

    def flush_run(values: list[int]) -> None:
        for i in range(0, len(values), stride):
            chunk = values[i:i + stride]
            anchors.append(int(chunk[len(chunk) // 2]))

    for step in unique:
        if not run or step == run[-1] + 1:
            run.append(step)
            continue
        flush_run(run)
        run = [step]
    if run:
        flush_run(run)

    return np.asarray(anchors, dtype=np.int32)


def quantize_path_progress_candidate_step(candidates: list[int], stride: int, raw_step: int) -> int:
    """Map a raw candidate step to the median anchor of its stride chunk."""
    if not candidates:
        raise ValueError("No path-progress candidates available")
    stride = max(1, int(stride))
    raw_step = int(raw_step)
    unique = sorted(set(int(k) for k in candidates))
    if raw_step not in set(unique):
        raise ValueError(f"raw_step={raw_step} is not a path-progress candidate")
    if stride == 1:
        return raw_step

    run: list[int] = []

    def maybe_quantize(values: list[int]) -> Optional[int]:
        for i in range(0, len(values), stride):
            chunk = values[i:i + stride]
            if raw_step in chunk:
                return int(chunk[len(chunk) // 2])
        return None

    for step in unique:
        if not run or step == run[-1] + 1:
            run.append(step)
            continue
        anchor = maybe_quantize(run)
        if anchor is not None:
            return anchor
        run = [step]
    anchor = maybe_quantize(run)
    if anchor is not None:
        return anchor
    raise ValueError(f"raw_step={raw_step} was not found in candidate chunks")


def choose_raw_then_binned_candidate(candidates: list[int], stride: int, rng: np.random.Generator) -> int:
    """Sample the original raw candidate distribution, then quantize to an anchor."""
    if not candidates:
        raise ValueError("No path-progress candidates available")
    idx = int(rng.integers(0, len(candidates)))
    raw_step = int(candidates[idx])
    return quantize_path_progress_candidate_step(candidates, stride, raw_step)


def select_multi_binned_candidate_steps(candidates: list[int], stride: int, max_pulses: int) -> np.ndarray:
    """Select up to max_pulses coarse anchors from binned path-risk candidates."""
    anchors = bin_path_progress_candidate_steps(candidates, stride)
    max_pulses = max(1, int(max_pulses))
    if len(anchors) <= max_pulses:
        return anchors
    idxs = np.rint(np.linspace(0, len(anchors) - 1, max_pulses)).astype(int)
    idxs = np.unique(idxs)
    while len(idxs) < max_pulses:
        for idx in range(len(anchors)):
            if idx not in set(int(i) for i in idxs):
                idxs = np.sort(np.append(idxs, idx))
                break
    return anchors[np.sort(idxs[:max_pulses])]


def quantize_topk_candidate_step(raw_step: int, stride: int, effective_horizon: int) -> int:
    """Quantize a top-k risk point to a coarse stride anchor."""
    stride = max(1, int(stride))
    raw_step = max(1, int(raw_step))
    latest = max(1, int(effective_horizon) - 1)
    if stride == 1:
        return min(raw_step, latest)
    anchor = int(((raw_step + stride // 2) // stride) * stride)
    return int(np.clip(anchor, 1, latest))


def select_topk_binned_candidate_steps(
    dyn_traj: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    arrival_steps: int,
    effective_horizon: int,
    distance_threshold: float,
    stride: int,
    top_k: int,
    min_separation_steps: int,
    anchor_lead_steps: int = 0,
) -> np.ndarray:
    """Select local nearest-risk anchors under a separate candidate threshold."""
    distances = compute_path_progress_min_distances(
        dyn_traj=dyn_traj,
        start_xy=start_xy,
        goal_xy=goal_xy,
        arrival_steps=arrival_steps,
        effective_horizon=effective_horizon,
    )
    if distances.size == 0:
        return np.asarray([], dtype=np.int32)

    threshold = float(distance_threshold)
    valid = np.flatnonzero(np.isfinite(distances) & (distances < threshold))
    if valid.size == 0:
        return np.asarray([], dtype=np.int32)

    local_minima: list[int] = []
    for idx in valid.tolist():
        left = distances[idx - 1] if idx > 0 else np.inf
        right = distances[idx + 1] if idx + 1 < distances.size else np.inf
        cur = distances[idx]
        if cur <= left and cur <= right and (cur < left or cur < right):
            local_minima.append(int(idx))

    ordered = sorted(
        set(local_minima) or set(int(x) for x in valid.tolist()),
        key=lambda step: (float(distances[step]), int(step)),
    )
    top_k = max(1, int(top_k))
    min_sep = max(0, int(min_separation_steps))
    selected: list[int] = []
    for step in ordered:
        if all(abs(int(step) - int(prev)) >= min_sep for prev in selected):
            selected.append(int(step))
            if len(selected) >= top_k:
                break

    if not selected:
        return np.asarray([], dtype=np.int32)

    lead = max(0, int(anchor_lead_steps))
    anchors = [
        quantize_topk_candidate_step(step - lead, stride=stride, effective_horizon=distances.size)
        for step in selected
    ]
    return np.asarray(sorted(set(int(x) for x in anchors)), dtype=np.int32)


def actual_clearance_penalty(
    min_actual_distance: float,
    *,
    margin: float,
    weight: float,
) -> float:
    weight = max(0.0, float(weight))
    margin = max(1e-6, float(margin))
    if weight <= 0.0 or not np.isfinite(float(min_actual_distance)):
        return 0.0
    if float(min_actual_distance) >= margin:
        return 0.0
    violation = (margin - max(0.0, float(min_actual_distance))) / margin
    return float(-weight * violation * violation)


class USVEnvMinimal(gymnasium.Env):
    """Minimal USV environment with optional CI-VO/RC-COLREGs audit hooks."""

    POLICY_DYN_SLOTS = 6
    _class_verified: bool = False  # class-level flag to avoid per-subprocess spam

    def __init__(
        self,
        grid_map: Optional[np.ndarray] = None,
        scenario_dir: Optional[str | pathlib.Path] = None,
        load_on_reset: bool = False,
        max_episode_steps: int = 1000,
        dt: float = 0.1,
        render_freq: int = 5,
        fixed_goal: Optional[np.ndarray] = None,
        fixed_initial_position: Optional[np.ndarray] = None,
        fixed_initial_psi: Optional[float] = None,
        render_mode: bool = False,
        dynamic_obstacles: bool = False,
        init_goal_threshold: float = 1.0,
        init_collision_threshold: float = 1.6,
        use_filter: bool = True,
        # Burst mismatch config
        burst_enabled: bool = True,
        burst_episode_prob: float = 0.4,
        burst_start_mode: str = "mixed",
        burst_duration_steps: int = 60,
        measurement_cov_scale: float = 100.0,
        measurement_delay_steps: int = 0,
        bias_enabled: bool = True,
        bias_position_y: float = 0.75,
        bias_duration_steps: int = 30,
        nominal_position_std: float = 0.10,
        nominal_velocity_std: float = 0.03,
        path_progress_arrival_steps: int = 230,
        path_progress_candidate_stride: int = 1,
        path_progress_max_pulses: int = 1,
        path_progress_topk: int = 1,
        path_progress_min_separation_steps: int = 60,
        path_progress_anchor_lead_steps: int = 0,
        path_progress_candidate_distance_threshold: Optional[float] = None,
        kf_cache_dir: Optional[str | pathlib.Path] = None,
        kf_cache_mode: str = "off",
        filter_execution_mode: str = "precomputed",
        # Trust config
        trust_mode: str = "innovation_estimated",
        trust_window_size: int = 20,
        trust_t_min: float = 0.05,
        trust_t_max: float = 1.0,
        trust_aggregate: str = "min_over_risk_active",
        trust_innovation_mode: str = "position_only_2d",
        risk_distance_threshold: float = 8.0,
        tcpa_horizon: float = 10.0,
        # Reward config
        success_bonus: float = 100.0,
        collision_penalty: float = -300.0,
        timeout_penalty: float = -50.0,
        progress_weight: float = 30.0,
        cte_penalty_weight: float = 0.1,
        time_penalty: float = -0.01,
        near_risk_weight: float = 3.0,
        actual_clearance_weight: float = 0.0,
        actual_clearance_margin: float = 2.0,
        # Optional CI-VO shield config. Disabled by default.
        civo_enabled: bool = False,
        civo_confidence: float = 0.99,
        civo_shield_enabled: bool = False,
        civo_shield_distance: float = 10.0,
        civo_shield_tcpa_horizon: float = 12.0,
        civo_shield_method: str = "corecbf",
        civo_shield_gate_mode: str = "distance_tcpa",
        vo_cbf_alpha_vo: float = 10.0,
        vo_cbf_alpha_c: float = 10.0,
        vo_cbf_k_u: float = 2.0,
        vo_cbf_k_vo: float = 50.0,
        corecbf_variant: str = "deterministic",
        corecbf_colregs_reference_scale: float = 0.1875,
        corecbf_surge_accel: float = 1.0,
        corecbf_turn_accel: float = 1.0,
        corecbf_turn_direction: int = -1,
        corecbf_gain: float = 1.0,
        corecbf_safety_distance: float = 2.0,
        corecbf_tau_u_weight: float = 1.0,
        corecbf_tau_r_weight: float = 1.0,
        corecbf_separable_slack_enabled: bool = False,
        corecbf_slack_weight: float = 1.0e4,
        corecbf_osqp_max_iter: int = 4000,
        corecbf_osqp_eps_abs: float = 1e-5,
        corecbf_osqp_eps_rel: float = 1e-5,
        corecbf_osqp_polishing: bool = True,
        rc_colregs_enabled: bool = False,
        rc_colregs_reward_weight: float = 0.0,
        rc_colregs_d_safe: float = 3.0,
        rc_colregs_tau: float = 10.0,
        rc_colregs_kappa: float = 60.0,
        rc_colregs_kappa_beta: Optional[float] = None,
        rc_colregs_kappa_time: Optional[float] = None,
        actuator_tau_u_dot_max: Optional[float] = None,
        actuator_n_dot_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        retired_corecbf_args = sorted(
            {
                "civo_eta",
                "civo_qp_solver",
                "civo_hocbf_gamma",
                "civo_hocbf_margin_gate",
                "civo_recovery_accel",
                "civo_hocbf_qp_enabled",
                "civo_qp_slack_weight",
                "civo_qp_dot_tau_u_weight",
                "civo_qp_tau_r_weight",
                "civo_min_surge_action",
                "civo_nominal_min_surge_action",
                "civo_hocbf_gain",
                "civo_barrier_distance",
            }.intersection(kwargs)
        )
        if retired_corecbf_args:
            raise TypeError(
                f"retired CoReCBF argument(s): {', '.join(retired_corecbf_args)}; "
                "use the corecbf_* parameters for the first-order hard QP"
            )
        if "vo_cbf_prediction_steps" in kwargs:
            raise TypeError(
                "retired CBF-VO argument vo_cbf_prediction_steps; "
                "CBF-VO is fixed to one step"
            )

        # ---- Scenario files ----
        self.scenario_dir = pathlib.Path(scenario_dir) if scenario_dir else None
        self.load_on_reset = load_on_reset
        self._scenario_files = sorted(self.scenario_dir.glob("*.npz")) if self.scenario_dir else []
        self._current_scenario_path: Optional[pathlib.Path] = None
        self._sc_idx = 0
        self._shuffled_indices = None
        self._seq_pointer = 0

        # Count max dynamic obstacles
        self.DYN_MAX = 0
        if dynamic_obstacles and self._scenario_files:
            for p in self._scenario_files:
                with np.load(p, allow_pickle=True) as data:
                    if "dyn_traj" in data:
                        dt_arr = data["dyn_traj"]
                        n = 1 if dt_arr.ndim == 2 else dt_arr.shape[1]
                        self.DYN_MAX = max(self.DYN_MAX, n)
        if not dynamic_obstacles:
            self.DYN_MAX = 0
        self.DYN_MAX = max(self.DYN_MAX, 6)

        self.render_freq = render_freq
        if grid_map is None and not self._scenario_files:
            raise ValueError("Either `grid_map` or `scenario_dir` must be provided.")
        self.grid_map = grid_map
        self.H, self.W = self.grid_map.shape

        # ---- Basic params ----
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.fixed_initial_position = fixed_initial_position
        self.initial_position = np.array([self.W - 2, self.H - 2], dtype=np.float32)
        self.fixed_goal = fixed_goal
        self.goal = np.array([self.W - 3, self.H - 3], dtype=np.float32)
        self.fixed_initial_psi = fixed_initial_psi
        self.render_mode = render_mode
        self.dynamic_obstacles = dynamic_obstacles
        self.R_usv = 1.0
        self.path_progress_arrival_steps = int(path_progress_arrival_steps)
        self.path_progress_candidate_stride = max(1, int(path_progress_candidate_stride))
        self.path_progress_max_pulses = max(1, int(path_progress_max_pulses))
        self.path_progress_topk = max(1, int(path_progress_topk))
        self.path_progress_min_separation_steps = max(0, int(path_progress_min_separation_steps))
        self.path_progress_anchor_lead_steps = max(0, int(path_progress_anchor_lead_steps))
        self.path_progress_candidate_distance_threshold = (
            None
            if path_progress_candidate_distance_threshold is None
            else float(path_progress_candidate_distance_threshold)
        )
        self.forced_burst_start_step: Optional[int] = None
        if isinstance(kf_cache_dir, (list, tuple)):
            self.kf_cache_dirs = [pathlib.Path(p) for p in kf_cache_dir if p]
        elif kf_cache_dir:
            self.kf_cache_dirs = [pathlib.Path(kf_cache_dir)]
        else:
            self.kf_cache_dirs = []
        self.kf_cache_dir = self.kf_cache_dirs[0] if self.kf_cache_dirs else None
        self.kf_cache_mode = str(kf_cache_mode)
        valid_cache_modes = {"off", "read", "write", "read_write", "read_strict"}
        if self.kf_cache_mode not in valid_cache_modes:
            raise ValueError(f"kf_cache_mode must be one of {sorted(valid_cache_modes)}, got {self.kf_cache_mode!r}")
        if self.kf_cache_mode == "read_strict" and not self.kf_cache_dirs:
            raise ValueError("read_strict requires kf_cache_dir")
        valid_filter_execution_modes = {"precomputed", "online_exact"}
        self.filter_execution_mode = str(filter_execution_mode)
        if self.filter_execution_mode not in valid_filter_execution_modes:
            raise ValueError(
                "filter_execution_mode must be one of "
                f"{sorted(valid_filter_execution_modes)}"
            )
        if (
            self.filter_execution_mode == "online_exact"
            and self.kf_cache_mode != "read_strict"
        ):
            raise ValueError(
                "online_exact requires kf_cache_mode='read_strict'"
            )
        self._kf_cache_hit = False
        self._kf_cache_path: Optional[pathlib.Path] = None
        self._kf_reference_hat: Optional[np.ndarray] = None
        self._kf_reference_tk: Optional[np.ndarray] = None
        self._kf_reference_p: Optional[np.ndarray] = None
        self._online_filter_rngs: list[np.random.Generator] = []
        self._online_filter_x: list[np.ndarray] = []
        self._online_filter_p: list[np.ndarray] = []
        self._online_filter_gt: list[np.ndarray] = []
        self._online_filter_model: Optional[tuple[np.ndarray, ...]] = None
        self._online_cache_checks = 0
        self._online_cache_mismatches = 0
        self._last_policy_obstacle_ids: tuple[int, ...] = ()

        # ---- Dynamic obstacles ----
        self.dyn_obs_num = 0
        self.dyn_radius = 1.0
        self.dyn_traj: Optional[np.ndarray] = None
        self.dyn_seeds: Optional[np.ndarray] = None
        self.dyn_pos = np.zeros((0, 5), dtype=np.float32)
        self.dyn_step: int = 0
        self.use_dyn_replay: bool = False

        # ---- Thresholds ----
        self.goal_threshold = init_goal_threshold
        self.collision_threshold = init_collision_threshold
        self.obs_norm_range = 32.0

        # ---- Reward config ----
        self.success_bonus = success_bonus
        self.collision_penalty = collision_penalty
        self.timeout_penalty = timeout_penalty
        self.progress_weight = progress_weight
        self.cte_penalty_weight = cte_penalty_weight
        self.time_penalty_per_step = time_penalty
        self.yaw_smooth_weight = 0.05          # r² penalty, was hardcoded 0.01
        self.near_risk_weight = near_risk_weight  # 1/dist penalty for close targets
        self.actual_clearance_weight = max(0.0, float(actual_clearance_weight))
        self.actual_clearance_margin = max(1e-6, float(actual_clearance_margin))
        self.safety_distance = 2.0             # same as collision radius

        # ---- Optional CI-VO shield config ----
        self.civo_enabled = bool(civo_enabled or civo_shield_enabled)
        self.civo_confidence = float(civo_confidence)
        if not math.isfinite(self.civo_confidence) or not 0.0 < self.civo_confidence < 1.0:
            raise ValueError("civo_confidence must be finite and strictly between 0 and 1")
        self._civo_sigma_scale = math.sqrt(-2.0 * math.log1p(-self.civo_confidence))
        self.civo_shield_enabled = bool(civo_shield_enabled)
        self.civo_shield_distance = max(0.0, float(civo_shield_distance))
        self.civo_shield_tcpa_horizon = max(0.0, float(civo_shield_tcpa_horizon))
        self.civo_shield_method = str(civo_shield_method)
        if self.civo_shield_method not in {
            "corecbf",
            "relaxed_vo_cbf",
            "cbf_vo",
        }:
            raise ValueError(
                "civo_shield_method must be 'corecbf', 'relaxed_vo_cbf', "
                "or 'cbf_vo'"
            )
        self.civo_shield_gate_mode = str(civo_shield_gate_mode)
        if self.civo_shield_gate_mode not in {
            "native",
            "all_obstacles",
            "distance_tcpa",
        }:
            raise ValueError(
                "civo_shield_gate_mode must be 'native', 'all_obstacles', "
                "or 'distance_tcpa'"
            )
        self.vo_cbf_alpha_vo = float(vo_cbf_alpha_vo)
        self.vo_cbf_alpha_c = float(vo_cbf_alpha_c)
        self.vo_cbf_k_u = float(vo_cbf_k_u)
        self.vo_cbf_k_vo = float(vo_cbf_k_vo)
        vo_values = np.array(
            [
                self.vo_cbf_alpha_vo,
                self.vo_cbf_alpha_c,
                self.vo_cbf_k_u,
                self.vo_cbf_k_vo,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(vo_values)) or np.any(vo_values <= 0.0):
            raise ValueError(
                "all Relaxed VO-CBF gains and weights must be "
                "positive and finite"
            )
        self.corecbf_variant = str(corecbf_variant)
        if self.corecbf_variant not in {"deterministic", "credibility_colregs"}:
            raise ValueError("corecbf_variant must be 'deterministic' or 'credibility_colregs'")
        if self.corecbf_variant == "credibility_colregs" and not use_filter:
            raise ValueError("credibility_colregs requires use_filter=True for Pf and oracle Pm")
        try:
            self.corecbf_colregs_reference_scale = float(
                corecbf_colregs_reference_scale
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "corecbf_colregs_reference_scale must be finite and lie in [0, 1]"
            ) from exc
        if (
            not math.isfinite(self.corecbf_colregs_reference_scale)
            or not 0.0 <= self.corecbf_colregs_reference_scale <= 1.0
        ):
            raise ValueError(
                "corecbf_colregs_reference_scale must be finite and lie in [0, 1]"
            )
        self.corecbf_surge_accel = float(corecbf_surge_accel)
        self.corecbf_turn_accel = float(corecbf_turn_accel)
        self.corecbf_gain = float(corecbf_gain)
        self.corecbf_safety_distance = float(corecbf_safety_distance)
        self.corecbf_tau_u_weight = float(corecbf_tau_u_weight)
        self.corecbf_tau_r_weight = float(corecbf_tau_r_weight)
        if type(corecbf_separable_slack_enabled) is not bool:
            raise ValueError("corecbf_separable_slack_enabled must be boolean")
        self.corecbf_separable_slack_enabled = corecbf_separable_slack_enabled
        self.corecbf_slack_weight = float(corecbf_slack_weight)
        corecbf_values = np.array(
            [
                self.corecbf_surge_accel,
                self.corecbf_turn_accel,
                self.corecbf_gain,
                self.corecbf_safety_distance,
                self.corecbf_tau_u_weight,
                self.corecbf_tau_r_weight,
                self.corecbf_slack_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(corecbf_values)) or np.any(corecbf_values <= 0.0):
            raise ValueError("all corecbf acceleration, gain, distance, and weight parameters must be positive and finite")
        if corecbf_turn_direction not in (-1, 1):
            raise ValueError("corecbf_turn_direction must be fixed at -1 or 1")
        self.corecbf_turn_direction = int(corecbf_turn_direction)
        if self.corecbf_safety_distance < self.R_usv + self.dyn_radius:
            raise ValueError("corecbf_safety_distance must cover the combined physical radius")
        self.corecbf_osqp_max_iter = max(1, int(corecbf_osqp_max_iter))
        self.corecbf_osqp_eps_abs = max(1e-9, float(corecbf_osqp_eps_abs))
        self.corecbf_osqp_eps_rel = max(1e-9, float(corecbf_osqp_eps_rel))
        self.corecbf_osqp_polishing = bool(corecbf_osqp_polishing)
        self.actuator_tau_u_dot_max = (
            None if actuator_tau_u_dot_max is None else float(actuator_tau_u_dot_max)
        )
        if self.actuator_tau_u_dot_max is not None and (
            not math.isfinite(self.actuator_tau_u_dot_max)
            or self.actuator_tau_u_dot_max <= 0.0
        ):
            raise ValueError("actuator_tau_u_dot_max must be a positive finite value")
        self.actuator_n_dot_max = None if actuator_n_dot_max is None else max(0.0, float(actuator_n_dot_max))
        if self.actuator_tau_u_dot_max is not None and self.actuator_n_dot_max:
            raise ValueError("actuator_tau_u_dot_max and actuator_n_dot_max cannot both be enabled")
        if self.civo_shield_enabled and self.actuator_n_dot_max:
            raise ValueError("the safety filters do not include actuator rate constraints")
        if (
            self.civo_shield_enabled
            and self.actuator_tau_u_dot_max is not None
            and self.civo_shield_method != "cbf_vo"
        ):
            raise ValueError("the safety filters do not include actuator rate constraints")
        self.civo_cones: dict[int, dict] = {}
        self._last_civo_shield_changed = False
        self._last_civo_shield_delta = 0.0
        self._last_civo_shield_latency_ms = 0.0
        self._last_civo_shield_latency_ns = 0
        self._last_actuator_mapping_ns = 0
        self._last_online_filter_timing_ns = {
            "estimator_step": 0,
            "measurement_sim_ns": 0,
            "filter_ns": 0,
            "trust_ns": 0,
        }
        self._last_coordinate_ns = 0
        self._last_safety_precompute_ns = 0
        self._last_observation_build_ns = 0
        self._prepared_control_latency_ns: dict[str, int] = {}
        self._last_civo_qp_success = False
        self._last_civo_qp_solver = "none"
        self._last_civo_qp_status = ""
        self._last_civo_qp_iter = 0
        self._last_civo_constraint_count = 0
        self._last_civo_obstacle_id = -1
        self._last_civo_h = float("nan")
        self._last_civo_residual = float("nan")
        self._last_civo_slack = 0.0
        self._last_civo_prop_override: Optional[tuple[float, float]] = None
        self._last_civo_mechanism = _empty_vo_cbf_diagnostics()
        self._last_credible_geometry_fallback_ids: tuple[int, ...] = ()
        self._last_credible_geometry_fallback_reasons: tuple[str, ...] = ()
        self._credible_geometry_fallback_steps = 0
        self._credible_geometry_fallback_obstacle_steps = 0
        self._last_tau_u_cmd = 0.0
        self._last_n_port_cmd = 0.0
        self._last_n_stbd_cmd = 0.0
        self._civo_ep_steps = 0
        self._vo_cbf_ep_fallback_count = 0
        self._vo_cbf_ep_domain_exit_count = 0
        self._civo_ep_active_steps = 0
        self._civo_ep_shield_steps = 0
        self._civo_ep_delta_sum = 0.0
        self._civo_ep_delta_max = 0.0
        self._civo_ep_slack_sum = 0.0
        self._civo_ep_slack_max = 0.0
        self._civo_ep_latency_ms: list[float] = []

        # Optional RC-COLREGs metrics/reward. Disabled by default so old
        # checkpoints keep the same observation and reward surface.
        self.rc_colregs_enabled = bool(
            rc_colregs_enabled or self.corecbf_variant == "credibility_colregs"
        )
        self.rc_colregs_reward_weight = max(0.0, float(rc_colregs_reward_weight))
        self.rc_colregs_theta_h = float(np.deg2rad(5.0))
        self.rc_colregs_theta_lim = float(np.deg2rad(112.5))
        self.rc_colregs_kappa = max(1e-6, float(rc_colregs_kappa))
        self.rc_colregs_kappa_beta = max(
            1e-6,
            float(rc_colregs_kappa if rc_colregs_kappa_beta is None else rc_colregs_kappa_beta),
        )
        self.rc_colregs_kappa_time = max(
            1e-6,
            float(rc_colregs_kappa if rc_colregs_kappa_time is None else rc_colregs_kappa_time),
        )
        self.rc_colregs_d_safe = max(1e-6, float(rc_colregs_d_safe))
        self.rc_colregs_tau = max(1e-6, float(rc_colregs_tau))
        self.rc_colregs_required_starboard = 0.10
        self.rc_colregs_starboard_yaw_sign = -1.0
        self.colregs_phi_by_id: dict[int, float] = {}
        self.colregs_beta_by_id: dict[int, float] = {}
        self._last_colregs_aggregate_duty = 0.0
        self._last_colregs_reference_control = np.zeros(2, dtype=float)
        self._colregs_ep_required = 0
        self._colregs_ep_compliant = 0
        self._colregs_ep_active = 0

        # ---- Observation space ----
        state_low_9 = np.array([-1, -1, -1, -1, -1, -1, -1, 0, -1], dtype=np.float32)
        state_high_9 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        wall_low = np.zeros(4, dtype=np.float32)
        wall_high = np.ones(4, dtype=np.float32)
        state_low = np.concatenate([state_low_9, wall_low]).astype(np.float32)
        state_high = np.concatenate([state_high_9, wall_high]).astype(np.float32)
        state_space = spaces.Box(low=state_low, high=state_high, shape=(13,), dtype=np.float32)

        dyn_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.POLICY_DYN_SLOTS * 7,),
            dtype=np.float32,
        )
        dyn_mask_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.POLICY_DYN_SLOTS,),
            dtype=np.float32,
        )

        # Observation contains state + dynamic target slots + their validity mask.
        # t_min remains in info only.
        self.observation_space = spaces.Dict({
            "state": state_space,
            "dyn": dyn_space,
            "dyn_mask": dyn_mask_space,
        })

        # ---- Action space ----
        self.Vm = 3.0864
        self.u_max, self.v_max, self.r_max = 3.0864, 1.0, 1.5
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # ---- USV dynamics params ----
        self.usv_params = {
            "m11": 68.448727294, "m22": 155.0, "m33": 41.85,
            "x_u": -77.554432348, "y_v": -155.0, "n_r": -41.85,
            "yaw_nonlinear": 10.0,
            "k_pos": 0.01108, "k_neg": 0.006445, "y_pontoon": 0.395,
            "n_max": 103.930864274, "n_min": -101.736665504,
            "u_max": 10.0, "v_max": 5.0, "r_max": 3.0,
        }

        # ---- State cache ----
        self.ship_state = np.zeros(6, dtype=np.float32)
        self.obstacle_coords = np.argwhere(self.grid_map == 1)
        self.pre_distance_to_target = 0.0

        # ---- Rendering ----
        self.fig, self.ax = None, None
        self.path = []

        # ---- Filter ----
        self.obstacle_estimates = {}
        self.dyn_hat_world = []
        self.dyn_tk = []
        self.dyn_P_world = []
        self.use_filter = use_filter
        self.dyn_true_traj = [[] for _ in range(self.dyn_obs_num)]
        self.dyn_est_traj = [[] for _ in range(self.dyn_obs_num)]
        self._last_t_min = 1.0

        # ---- Trust factor ----
        self.trust_mode = trust_mode
        if trust_mode == "oracle_tmse":
            self.trust_computer = OracleTrust(
                t_min=trust_t_min,
                t_max=trust_t_max,
                risk_distance_threshold=risk_distance_threshold,
                tcpa_horizon=tcpa_horizon,
                aggregate=trust_aggregate,
                norm_type="spectral_2",
            )
        else:
            self.trust_computer = InnovationConsistencyTrust(
                window_size=trust_window_size,
                t_min=trust_t_min,
                t_max=trust_t_max,
                risk_distance_threshold=risk_distance_threshold,
                tcpa_horizon=tcpa_horizon,
                aggregate=trust_aggregate,
                innovation_mode=trust_innovation_mode,
            )

        # ---- Burst mismatch injector ----
        self.noise_injector = BurstMismatchInjector(
            burst_enabled=burst_enabled,
            burst_episode_prob=burst_episode_prob,
            burst_duration_steps=burst_duration_steps,
            measurement_cov_scale=measurement_cov_scale,
            bias_enabled=bias_enabled,
            bias_position_y=bias_position_y,
            bias_duration_steps=bias_duration_steps,
            start_mode=burst_start_mode,
            measurement_delay_steps=measurement_delay_steps,
            risk_distance_threshold=risk_distance_threshold,
            tcpa_horizon=tcpa_horizon,
            nominal_position_std=nominal_position_std,
            nominal_velocity_std=nominal_velocity_std,
        )

        # ---- RNG ----
        self.rng = np.random.default_rng()

        # ---- Scenario length tracking ----
        self.scenario_T: int = 0  # Set when dyn_traj is loaded
        self._ground_truth_fallback_used: bool = False

        # ---- Verification (class-level to avoid subprocess spam) ----

        # ---- Colors ----
        self._init_dyn_colors()

    # ==================================================================
    #   Observation
    # ==================================================================

    def _get_obs(self) -> dict:
        """Build observation dict. t_min is in info, not observation."""
        psi = self.ship_state[2]
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        u, v, r = self.ship_state[3:6]
        u_norm = np.clip(u / (self.u_max + 1e-6), -1.0, 1.0)
        v_norm = np.clip(v / (self.v_max + 1e-6), -1.0, 1.0)
        r_norm = np.clip(r / (self.r_max + 1e-6), -1.0, 1.0)
        dx = (self.goal[0] - self.ship_state[0]) / self.W
        dy = (self.goal[1] - self.ship_state[1]) / self.H
        rho = np.hypot(dx, dy)
        rho_norm = np.clip(rho, 0.0, 1.0)
        theta = np.arctan2(dy, dx) - psi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        theta_norm = theta / np.pi
        full_state = np.array(
            [sin_psi, cos_psi, u_norm, v_norm, r_norm, dx, dy, rho_norm, theta_norm],
            dtype=np.float32,
        )

        # Wall distances
        x = float(np.clip(self.ship_state[0], 0.0, self.W - 1.0))
        y = float(np.clip(self.ship_state[1], 0.0, self.H - 1.0))
        wall_obs = np.array([
            x / max(self.W - 1.0, 1.0),
            (self.W - 1.0 - x) / max(self.W - 1.0, 1.0),
            y / max(self.H - 1.0, 1.0),
            (self.H - 1.0 - y) / max(self.H - 1.0, 1.0),
        ], dtype=np.float32)
        full_state = np.concatenate([full_state, wall_obs]).astype(np.float32)

        # Dynamic obstacle features (7-dim each)
        est_items = []
        for i, est in self.obstacle_estimates.items():
            ex, ey = float(est["dx"]), float(est["dy"])
            dist = math.hypot(ex, ey)
            est_items.append((dist, int(i), est))
        est_items.sort(key=lambda t: (t[0], t[1]))
        est_items = est_items[:self.POLICY_DYN_SLOTS]
        self._last_policy_obstacle_ids = tuple(
            obstacle_id for _, obstacle_id, _ in est_items
        )

        dyn_feat_list = []
        dyn_mask_list = []
        pos_scale = max(self.obs_norm_range, 1e-6)
        v_scale = max(self.u_max, 1e-6)

        for _, i, est in est_items:
            x_loc = float(est.get("dx", 0.0))
            y_loc = float(est.get("dy", 0.0))
            x_tilde = np.clip(x_loc / pos_scale, -1.0, 1.0)
            y_tilde = np.clip(y_loc / pos_scale, -1.0, 1.0)

            vx_loc = float(est.get("vx", 0.0))
            vy_loc = float(est.get("vy", 0.0))
            vx_n = np.clip(vx_loc / v_scale, -1.0, 1.0)
            vy_n = np.clip(vy_loc / v_scale, -1.0, 1.0)

            rho_loc = float(math.hypot(x_loc, y_loc))
            sin_rel = y_loc / rho_loc if rho_loc > 1e-6 else 0.0
            cos_rel = x_loc / rho_loc if rho_loc > 1e-6 else 1.0

            # Per-obstacle trust (from innovation consistency)
            t_i = float(est.get("trust", 1.0))
            t_i = float(np.clip(t_i, 0.0, 1.0))

            dyn_feat_list.extend([x_tilde, y_tilde, vx_n, vy_n, sin_rel, cos_rel, t_i])
            dyn_mask_list.append(1.0)

        # Pad inactive slots with zeros; dyn_mask tells the policy which slots are real.
        while len(dyn_feat_list) < self.POLICY_DYN_SLOTS * 7:
            dyn_feat_list.extend([0.0] * 7)
            dyn_mask_list.append(0.0)

        dyn_feat = np.array(dyn_feat_list, dtype=np.float32)
        dyn_mask = np.array(dyn_mask_list[:self.POLICY_DYN_SLOTS], dtype=np.float32)

        # Global trust: min over risk-active targets
        t_global = self._compute_global_trust()
        self._last_t_min = t_global

        # Store trust for info output (NOT in observation)
        self._last_t_min = t_global

        return {
            "state": full_state,
            "dyn": dyn_feat,
            "dyn_mask": dyn_mask,
        }

    def _prepare_safety_inputs(self) -> None:
        started = time.perf_counter_ns()
        self._refresh_civo_cones()
        self._update_colregs_signal()
        elapsed = time.perf_counter_ns() - started
        self._last_safety_precompute_ns = (
            int(elapsed) if self.civo_shield_enabled else 0
        )

    def _build_control_observation(self) -> dict:
        started = time.perf_counter_ns()
        obs = self._get_obs()
        self._last_observation_build_ns = time.perf_counter_ns() - started
        filter_timing = self._last_online_filter_timing_ns
        self._prepared_control_latency_ns = {
            "estimator_step": int(self.dyn_step),
            "measurement_sim_ns": int(
                filter_timing["measurement_sim_ns"]
            ),
            "filter_ns": int(filter_timing["filter_ns"]),
            "trust_ns": int(filter_timing["trust_ns"]),
            "coordinate_ns": int(self._last_coordinate_ns),
            "safety_precompute_ns": int(
                self._last_safety_precompute_ns
            ),
            "observation_build_ns": int(
                self._last_observation_build_ns
            ),
        }
        return obs

    def prepared_control_latency_ns(self) -> dict[str, int]:
        if not self._prepared_control_latency_ns:
            raise RuntimeError("control input timing is not prepared")
        return dict(self._prepared_control_latency_ns)

    def _compute_global_trust(self) -> float:
        """Compute global trust by aggregating per-obstacle trust over
        risk-active targets."""
        u_own = float(self.ship_state[3])
        v_own = float(self.ship_state[4])
        return self.trust_computer.get_global_trust(
            self.obstacle_estimates, u_own, v_own,
        )

    # ==================================================================
    #   Reset
    # ==================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.seed(seed)
            if self._scenario_files:
                num_files = len(self._scenario_files)
                indices = np.arange(num_files)
                self.np_random.shuffle(indices)
                self._shuffled_indices = indices
                self._seq_pointer = 0
                self._sc_idx = self._shuffled_indices[self._seq_pointer]
        elif self._scenario_files and self._shuffled_indices is not None:
            self._seq_pointer = (self._seq_pointer + 1) % len(self._scenario_files)
            self._sc_idx = self._shuffled_indices[self._seq_pointer]
        elif self._scenario_files and self._shuffled_indices is None:
            self._shuffled_indices = np.arange(len(self._scenario_files))
            self.np_random.shuffle(self._shuffled_indices)
            self._sc_idx = self._shuffled_indices[0]

        # Load scenario
        if self.load_on_reset and self._scenario_files:
            tried = 0
            loaded = False
            while tried < len(self._scenario_files) and not loaded:
                path = self._scenario_files[self._sc_idx]
                tried += 1
                try:
                    data = np.load(path, allow_pickle=True)
                    required = ["grid", "init_pos", "init_psi", "goal"]
                    if not all(k in data.files for k in required):
                        self._sc_idx = (self._sc_idx + 1) % len(self._scenario_files)
                        continue
                    self.set_grid(data["grid"])
                    self._current_scenario_path = pathlib.Path(path)
                    self.fixed_initial_position = data["init_pos"]
                    self.fixed_initial_psi = float(data["init_psi"])
                    self.fixed_goal = data["goal"]
                    if "dyn_seeds" in data:
                        self.dyn_seeds = data["dyn_seeds"]
                    else:
                        self.dyn_seeds = None
                    if self.dynamic_obstacles and "dyn_traj" in data.files:
                        dyn_traj_norm = self._normalize_dyn_traj(data["dyn_traj"])
                        if dyn_traj_norm is not None:
                            self.set_dyn_traj(dyn_traj_norm)
                            self.use_dyn_replay = True
                            self.dyn_hat_world = []
                            self.dyn_tk = []
                            self.dyn_P_world = []
                        else:
                            self.use_dyn_replay = False
                    else:
                        self.use_dyn_replay = False
                    loaded = True
                except DynamicObstacleCapacityError:
                    raise
                except Exception as e:
                    self._sc_idx = (self._sc_idx + 1) % len(self._scenario_files)
                    continue

        if self.fixed_initial_position is None:
            raise RuntimeError("init_pos not provided!")
        if self.fixed_goal is None:
            raise RuntimeError("goal not provided!")
        if self.fixed_initial_psi is None:
            raise RuntimeError("init_psi not provided!")

        x0, y0 = map(float, self.fixed_initial_position)
        psi0 = float(self.fixed_initial_psi)
        self.goal = self.fixed_goal.astype(np.float32)
        self.initial_position = np.array([x0, y0], dtype=np.float32)

        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.ship_state = np.array([x0, y0, psi0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.pre_distance_to_target = np.linalg.norm(self.goal - self.initial_position)
        self.path = [(x0, y0)]
        self.dyn_step = 0

        if self.use_dyn_replay and self.dyn_traj is not None:
            self.dyn_pos = self.dyn_traj[0].copy()
            for i in range(self.dyn_obs_num):
                self.dyn_true_traj[i].clear()
                self.dyn_est_traj[i].clear()
                self.dyn_true_traj[i].append(
                    np.array([float(self.dyn_pos[i, 0]), float(self.dyn_pos[i, 1])], dtype=float),
                )
        else:
            self.dyn_obs_num = 0
            self.dyn_pos = np.zeros((0, 5), dtype=np.float32)
            self.use_dyn_replay = False
            self.dyn_true_traj = []
            self.dyn_est_traj = []

        self.obstacle_estimates.clear()
        self.civo_cones.clear()
        self._last_civo_shield_changed = False
        self._last_civo_shield_delta = 0.0
        self._last_civo_shield_latency_ms = 0.0
        self._last_civo_shield_latency_ns = 0
        self._last_actuator_mapping_ns = 0
        self._last_online_filter_timing_ns = {
            "estimator_step": 0,
            "measurement_sim_ns": 0,
            "filter_ns": 0,
            "trust_ns": 0,
        }
        self._last_coordinate_ns = 0
        self._last_safety_precompute_ns = 0
        self._last_observation_build_ns = 0
        self._prepared_control_latency_ns = {}
        self._last_civo_qp_success = False
        self._last_civo_qp_solver = "none"
        self._last_civo_qp_status = ""
        self._last_civo_qp_iter = 0
        self._last_civo_constraint_count = 0
        self._last_civo_obstacle_id = -1
        self._last_civo_h = float("nan")
        self._last_civo_residual = float("nan")
        self._last_civo_slack = 0.0
        self._last_civo_prop_override = None
        self._last_civo_mechanism = _empty_vo_cbf_diagnostics()
        self._last_credible_geometry_fallback_ids = ()
        self._last_credible_geometry_fallback_reasons = ()
        self._credible_geometry_fallback_steps = 0
        self._credible_geometry_fallback_obstacle_steps = 0
        self._last_tau_u_cmd = 0.0
        self._last_n_port_cmd = 0.0
        self._last_n_stbd_cmd = 0.0
        self._civo_ep_steps = 0
        self._vo_cbf_ep_fallback_count = 0
        self._vo_cbf_ep_domain_exit_count = 0
        self._civo_ep_active_steps = 0
        self._civo_ep_shield_steps = 0
        self._civo_ep_delta_sum = 0.0
        self._civo_ep_delta_max = 0.0
        self._civo_ep_slack_sum = 0.0
        self._civo_ep_slack_max = 0.0
        self._civo_ep_latency_ms = []
        self.colregs_phi_by_id = {}
        self.colregs_beta_by_id = {}
        self._last_colregs_aggregate_duty = 0.0
        self._last_colregs_reference_control = np.zeros(2, dtype=float)
        self._colregs_ep_required = 0
        self._colregs_ep_compliant = 0
        self._colregs_ep_active = 0
        self.trust_computer.reset()
        self.noise_injector.reset_stats()

        # Noise injector: decide burst/bias for this episode
        T = int(self.dyn_traj.shape[0]) if self.dyn_traj is not None else 256
        effective_horizon = min(self.max_episode_steps, T)
        self.noise_injector.reset(self.rng, T, effective_horizon=effective_horizon)
        self._configure_path_progress_burst_from_replay()
        # Burst timing is episode-specific; filter replay must be resolved after
        # the final burst start is known, even when the scenario was preloaded.
        self.dyn_hat_world = []
        self.dyn_tk = []
        self.dyn_P_world = []
        self._kf_cache_hit = False
        self._kf_cache_path = None
        self._kf_reference_hat = None
        self._kf_reference_tk = None
        self._kf_reference_p = None
        self._online_filter_rngs = []
        self._online_filter_x = []
        self._online_filter_p = []
        self._online_filter_gt = []
        self._online_filter_model = None
        self._online_cache_checks = 0
        self._online_cache_mismatches = 0
        self._last_policy_obstacle_ids = ()

        self.pre_distance_to_target = distance_to_goal(self.ship_state, self.goal)
        self._ep_min_dcpa = float("inf")
        self._ep_min_actual_dist = float("inf")

        # Initial obstacle estimates
        if self.use_dyn_replay and self.dyn_traj is not None and self.dyn_obs_num > 0:
            if self.use_filter:
                self._filter_dynamic_obstacles()
            else:
                self._use_ground_truth_obstacles()
        self._prepare_safety_inputs()

        obs = self._build_control_observation()
        if self.filter_execution_mode == "online_exact":
            self._assert_online_cache_step(self.dyn_step)

        # Verification logging (once per class, not per subprocess)
        if not USVEnvMinimal._class_verified:
            USVEnvMinimal._class_verified = True
            print(f"[VERIFY] env obs = {list(self.observation_space.spaces.keys())}")
            civo_state = "ON" if self.civo_enabled else "OFF"
            shield_state = "ON" if self.civo_shield_enabled else "OFF"
            print(f"[VERIFY] t_min in INFO only | CI-VO={civo_state} | shield={shield_state}")

        return obs, {}

    # ==================================================================
    #   Step
    # ==================================================================

    def step(self, action):
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        action = self._shield_action_civo(policy_action)
        self.current_step += 1
        info: dict = {"step": self.current_step}

        policy_yaw = float(np.clip(policy_action[1], -1.0, 1.0))
        raw_surge = float(np.clip(action[0], -1.0, 1.0))
        raw_yaw = float(np.clip(action[1], -1.0, 1.0))
        info["action_before_env"] = policy_yaw
        info["civo_shield_changed"] = bool(self._last_civo_shield_changed)
        info["civo_shield_delta"] = float(self._last_civo_shield_delta)
        info["civo_shield_latency_ms"] = float(self._last_civo_shield_latency_ms)
        info["civo_qp_success"] = bool(self._last_civo_qp_success)
        info["civo_qp_solver"] = str(self._last_civo_qp_solver)
        info["civo_qp_status"] = str(self._last_civo_qp_status)
        info["civo_qp_iter"] = int(self._last_civo_qp_iter)
        info["civo_constraint_count"] = int(self._last_civo_constraint_count)
        info["civo_obstacle_id"] = int(self._last_civo_obstacle_id)
        info["civo_cbf_h"] = float(self._last_civo_h)
        info["civo_cbf_residual"] = float(self._last_civo_residual)
        info["civo_qp_slack"] = float(self._last_civo_slack)
        info["civo_shield_method"] = self.civo_shield_method
        info["vo_cbf_fallback_ep_count"] = int(self._vo_cbf_ep_fallback_count)
        info["vo_cbf_domain_exit_ep_count"] = int(
            self._vo_cbf_ep_domain_exit_count
        )
        info.update(self._last_civo_mechanism)
        info["credible_geometry_fallback_count"] = len(
            self._last_credible_geometry_fallback_ids
        )
        info["credible_geometry_fallback_ids"] = self._last_credible_geometry_fallback_ids
        info["credible_geometry_fallback_reasons"] = (
            self._last_credible_geometry_fallback_reasons
        )
        colregs_pre = self._collect_colregs_metrics(raw_yaw)

        # Map to propeller speeds
        actuator_started = time.perf_counter_ns()
        n_max = float(self.usv_params["n_max"])
        n_min = float(self.usv_params["n_min"])
        n_common = (raw_surge * n_max if raw_surge >= 0 else raw_surge * abs(n_min))
        n_common *= 0.70  # cruise speed ~1.5 m/s
        n_diff_max = min(n_max, abs(n_min))
        n_diff = raw_yaw * n_diff_max
        if self._last_civo_prop_override is not None:
            n_port, n_stbd = self._last_civo_prop_override
            n_port = float(np.clip(n_port, n_min, n_max))
            n_stbd = float(np.clip(n_stbd, n_min, n_max))
        else:
            n_port = np.clip(n_common + n_diff, n_min, n_max)
            n_stbd = np.clip(n_common - n_diff, n_min, n_max)
        n_port, n_stbd, tau_u_cmd_raw, tau_r_cmd_raw, tau_u_rate_limit_active = (
            self._apply_tau_u_rate_limit(float(n_port), float(n_stbd))
        )
        n_port, n_stbd = self._apply_propeller_slew(n_port, n_stbd)
        last_tau_u_cmd = float(self._last_tau_u_cmd)
        thrust_port = float(prop_thrust(n_port, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])))
        thrust_stbd = float(prop_thrust(n_stbd, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])))
        tau_u_cmd = float(thrust_port + thrust_stbd)
        tau_r_cmd = float(float(self.usv_params["y_pontoon"]) * (thrust_port - thrust_stbd))
        dot_tau_u_cmd = float((tau_u_cmd - last_tau_u_cmd) / max(float(self.dt), 1e-9))
        self._last_tau_u_cmd = tau_u_cmd
        self._last_actuator_mapping_ns = (
            time.perf_counter_ns() - actuator_started
        )
        info["safety_filter_ns"] = int(
            self._last_civo_shield_latency_ns
        )
        info["actuator_mapping_ns"] = int(
            self._last_actuator_mapping_ns
        )

        # Physics
        self.ship_state = update_usv_full_model(
            self.ship_state, n_port, n_stbd, self.dt, self.usv_params,
        )
        self.ship_state[2] = normalize_angle_0_to_2pi(self.ship_state[2])

        x_new, y_new, psi_new, u_new, v_new, r_new = self.ship_state

        # Dynamic obstacle replay + filter
        if self.use_dyn_replay and self.dyn_traj is not None:
            if self.dyn_step + 1 < len(self.dyn_traj):
                self.dyn_step += 1
            self.dyn_pos = self.dyn_traj[self.dyn_step].copy()
            if self.use_filter:
                self._filter_dynamic_obstacles()
            else:
                self._use_ground_truth_obstacles()
            for i in range(self.dyn_obs_num):
                self.dyn_true_traj[i].append(
                    np.array([float(self.dyn_pos[i, 0]), float(self.dyn_pos[i, 1])], dtype=float),
                )

        self._prepare_safety_inputs()
        self.path.append((x_new, y_new))

        # ================================================================
        #   Reward computation
        # ================================================================
        reward = 0.0
        terminated = False
        truncated = False

        # 1) Progress toward goal
        cur_dist = distance_to_goal(self.ship_state, self.goal)
        approach_reward = self.progress_weight * (self.pre_distance_to_target - cur_dist)
        self.pre_distance_to_target = cur_dist
        reward += approach_reward
        info["r_approach"] = float(approach_reward)

        # 2) Goal reached
        if cur_dist < self.goal_threshold:
            reward += self.success_bonus
            terminated = True
            info["reason"] = "goal_reached"

        # 3) Collision detection
        is_collided = False
        collision_reason = None

        if not in_bounds(x_new, y_new, self.W, self.H):
            is_collided = True
            collision_reason = "out_of_bounds"

        if not is_collided:
            xi, yi = int(round(x_new)), int(round(y_new))
            if 0 <= xi < self.W and 0 <= yi < self.H:
                if self.grid_map[yi, xi] == 1:
                    is_collided = True
                    collision_reason = "static_obs"

        if not is_collided and self.dyn_obs_num > 0:
            dyn_xy = self.dyn_pos[:, :2]
            usv_xy = np.array([x_new, y_new])
            dists = np.linalg.norm(dyn_xy - usv_xy, axis=1)
            if np.any(dists < (self.R_usv + self.dyn_radius)):
                is_collided = True
                collision_reason = "dynamic_obs"

        if is_collided:
            reward += self.collision_penalty
            terminated = True
            info["reason"] = collision_reason or "collision"
            info["r_collision"] = float(self.collision_penalty)

        # 4) DCPA/TCPA-based risk penalty: penalize predicted close encounters.
        #    Uses current-step predicted DCPA and TCPA from KF estimates.
        #    Only activates when 0 < TCPA < tcpa_horizon and DCPA < warning_distance.
        r_near_risk = 0.0
        if self.dyn_obs_num > 0 and not is_collided:
            u_own = float(u_new)
            v_own = float(v_new)
            warning_dist = 4.0          # DCPA below this triggers penalty
            tcpa_horizon_risk = 12.0    # only penalize encounters within this TCPA window
            risk_w = self.near_risk_weight  # configurable weight
            for est in self.obstacle_estimates.values():
                dx = float(est.get("dx", 0.0))
                dy = float(est.get("dy", 0.0))
                vx_obs = float(est.get("vx", 0.0))
                vy_obs = float(est.get("vy", 0.0))
                vx_rel = vx_obs - u_own
                vy_rel = vy_obs - v_own
                v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
                if v_rel_sq < 1e-6:
                    continue
                tcpa = -(dx * vx_rel + dy * vy_rel) / v_rel_sq
                if tcpa <= 0.0 or tcpa > tcpa_horizon_risk:
                    continue
                # Predicted position at CPA
                cx = dx + vx_rel * tcpa
                cy = dy + vy_rel * tcpa
                dcpa = float(np.hypot(cx, cy))
                if dcpa >= warning_dist:
                    continue
                # Quadratic penalty: stronger for closer DCPA, weighted by TCPA urgency
                violation = (warning_dist - dcpa) / warning_dist  # 0 at DCPA=warning, 1 at DCPA=0
                tcp_weight = 0.5 + 0.5 * (1.0 - tcpa / tcpa_horizon_risk)  # 1.0 at TCPA=0, 0.5 at TCPA=horizon
                r_near_risk -= risk_w * (violation ** 2) * tcp_weight
        reward += r_near_risk
        info["r_near_risk"] = float(r_near_risk)

        r_colregs = -self.rc_colregs_reward_weight * float(colregs_pre["colregs_violation"])
        reward += r_colregs
        info.update(colregs_pre)
        info["r_colregs"] = float(r_colregs)
        self._colregs_ep_required += int(colregs_pre["colregs_required_count"])
        self._colregs_ep_compliant += int(colregs_pre["colregs_compliant_count"])
        self._colregs_ep_active += int(colregs_pre["colregs_active_count"])

        civo_metrics = self._collect_civo_metrics(float(u_new), float(v_new))
        info.update(civo_metrics)
        self._civo_ep_steps += 1
        self._civo_ep_delta_sum += float(civo_metrics.get("civo_delta_sum", 0.0))
        self._civo_ep_delta_max = max(
            self._civo_ep_delta_max,
            float(civo_metrics.get("civo_delta_max", 0.0)),
        )
        self._civo_ep_slack_sum += float(self._last_civo_slack)
        self._civo_ep_slack_max = max(self._civo_ep_slack_max, float(self._last_civo_slack))
        if int(civo_metrics.get("civo_active_count", 0)) > 0:
            self._civo_ep_active_steps += 1
        if self._last_civo_shield_changed:
            self._civo_ep_shield_steps += 1

        # 5) Time penalty (small, per-step)
        reward += self.time_penalty_per_step
        info["r_time"] = self.time_penalty_per_step

        # 6) Cross-track error penalty
        p_start = self.initial_position
        p_goal = self.goal
        p_curr = self.ship_state[:2]
        v_line = p_goal - p_start
        v_curr = p_curr - p_start
        line_len = float(np.linalg.norm(v_line))
        if line_len > 1e-3:
            cross_prod = v_line[0] * v_curr[1] - v_line[1] * v_curr[0]
            cte = abs(cross_prod) / line_len
        else:
            cte = 0.0
        r_cte = -self.cte_penalty_weight * cte
        reward += r_cte
        info["cte"] = float(cte)
        info["r_cte"] = float(r_cte)

        # 7) Yaw-rate smoothness
        r_smooth = -self.yaw_smooth_weight * float(r_new ** 2)
        reward += r_smooth
        info["r_smooth"] = float(r_smooth)

        # ---- Trust & noise diagnostics ----
        t_global = self._last_t_min
        is_burst_now = self.noise_injector.is_burst(self.current_step, self._check_any_risk_active())
        is_bias_now = self.noise_injector.is_bias(self.current_step)
        is_risk_active = self._check_any_risk_active()
        min_dcpa_now = self._compute_min_dcpa()

        info["t_min"] = float(t_global)
        info["burst_active"] = bool(is_burst_now)
        info["bias_active"] = bool(is_bias_now)
        info["risk_active"] = bool(is_risk_active)
        info["min_dcpa"] = float(min_dcpa_now)
        if min_dcpa_now < self._ep_min_dcpa:
            self._ep_min_dcpa = float(min_dcpa_now)

        # Actual distance to closest target (for strict_success / unsafe_near_miss)
        min_actual_dist = float("inf")
        closest_actual_port = 0.0
        closest_actual_forward = 0.0
        if self.dyn_obs_num > 0:
            dyn_xy = self.dyn_pos[:, :2]
            usv_xy = np.array([x_new, y_new])
            actual_dists = np.linalg.norm(dyn_xy - usv_xy, axis=1)
            closest_idx = int(np.argmin(actual_dists))
            rel = dyn_xy[closest_idx] - usv_xy
            c_psi = float(np.cos(psi_new))
            s_psi = float(np.sin(psi_new))
            closest_actual_forward = float(c_psi * rel[0] + s_psi * rel[1])
            closest_actual_port = float(-s_psi * rel[0] + c_psi * rel[1])
            min_actual_dist = float(np.min(actual_dists))
        info["min_actual_distance"] = min_actual_dist
        info["closest_actual_forward"] = float(closest_actual_forward)
        info["closest_actual_port"] = float(closest_actual_port)
        if min_actual_dist < self._ep_min_actual_dist:
            self._ep_min_actual_dist = float(min_actual_dist)
        r_actual_clearance = 0.0
        if not is_collided:
            r_actual_clearance = actual_clearance_penalty(
                min_actual_dist,
                margin=self.actual_clearance_margin,
                weight=self.actual_clearance_weight,
            )
            reward += r_actual_clearance
        info["r_actual_clearance"] = float(r_actual_clearance)
        info["risk_distance_threshold"] = float(self.noise_injector.risk_distance_threshold)
        info["tcpa_horizon"] = float(self.noise_injector.tcpa_horizon)

        self.noise_injector.update_stats(self.current_step, is_risk_active, t_global)

        # ---- Info ----
        info.update({
            "ship_xy": self.ship_state[:2].copy(),
            "ship_x": float(x_new),
            "ship_y": float(y_new),
            "ship_psi": float(psi_new),
            "ship_u": float(u_new),
            "ship_v": float(v_new),
            "ship_r": float(r_new),
            "goal_xy": self.goal.copy(),
            "reward_total": float(reward),
            "distance_to_goal": float(cur_dist),
            "action": action.tolist(),
            "policy_action": policy_action.tolist(),
            "n_port_cmd": float(n_port),
            "n_stbd_cmd": float(n_stbd),
            "tau_u_cmd_raw": float(tau_u_cmd_raw),
            "tau_r_cmd_raw": float(tau_r_cmd_raw),
            "tau_u_rate_limit_active": bool(tau_u_rate_limit_active),
            "tau_u_cmd": float(tau_u_cmd),
            "dot_tau_u_cmd": float(dot_tau_u_cmd),
            "tau_r_cmd": float(tau_r_cmd),
            "V": float(np.sqrt(u_new ** 2 + v_new ** 2)),
            "action_before_env": policy_yaw,
            "action_executed": raw_yaw,
        })

        # Timeout: max episode steps OR scenario trajectory exhausted
        timeout_reason = None
        if self.current_step >= self.max_episode_steps:
            truncated = True
            timeout_reason = "max_steps"
        elif self.use_dyn_replay and self.dyn_step >= self.scenario_T:
            truncated = True
            timeout_reason = "scenario_end"
        if timeout_reason:
            reward += self.timeout_penalty
            info["r_timeout"] = float(self.timeout_penalty)
            info["timeout_reason"] = timeout_reason

        # Episode-end diagnostic ratios
        if terminated or truncated:
            ratios = self.noise_injector.get_ratios()
            info.update(ratios)
            info["ground_truth_fallback_used"] = bool(self._ground_truth_fallback_used)
            info["scenario_T"] = int(self.scenario_T)
            info["ep_min_dcpa"] = float(self._ep_min_dcpa)
            info["ep_min_actual_distance"] = float(self._ep_min_actual_dist)
            info["collision_radius"] = float(self.R_usv + self.dyn_radius)
            civo_steps = max(1, int(self._civo_ep_steps))
            info["civo_ep_active_ratio"] = float(self._civo_ep_active_steps / civo_steps)
            info["civo_ep_shield_ratio"] = float(self._civo_ep_shield_steps / civo_steps)
            info["civo_ep_delta_mean"] = float(self._civo_ep_delta_sum / civo_steps)
            info["civo_ep_delta_max"] = float(self._civo_ep_delta_max)
            info["civo_ep_slack_mean"] = float(self._civo_ep_slack_sum / civo_steps)
            info["civo_ep_slack_max"] = float(self._civo_ep_slack_max)
            lat = np.asarray(self._civo_ep_latency_ms, dtype=float)
            info["civo_ep_latency_mean_ms"] = float(np.mean(lat)) if lat.size else 0.0
            info["civo_ep_latency_p95_ms"] = float(np.percentile(lat, 95)) if lat.size else 0.0
            info["civo_ep_latency_p99_ms"] = float(np.percentile(lat, 99)) if lat.size else 0.0
            info["civo_ep_latency_max_ms"] = float(np.max(lat)) if lat.size else 0.0
            info["credible_geometry_fallback_ep_steps"] = int(
                self._credible_geometry_fallback_steps
            )
            info["credible_geometry_fallback_ep_obstacle_steps"] = int(
                self._credible_geometry_fallback_obstacle_steps
            )
            info["colregs_ep_required"] = int(self._colregs_ep_required)
            info["colregs_ep_compliant"] = int(self._colregs_ep_compliant)
            info["colregs_ep_active"] = int(self._colregs_ep_active)
            info["colregs_ep_compliance"] = float(
                self._colregs_ep_compliant / self._colregs_ep_required
                if self._colregs_ep_required
                else 1.0
            )

        # Rendering
        if self.render_mode and self.current_step % self.render_freq == 0:
            self.render()

        obs = self._build_control_observation()
        if self.filter_execution_mode == "online_exact":
            self._assert_online_cache_step(self.dyn_step)
        info["cache_exact_match"] = bool(
            self.filter_execution_mode == "online_exact"
            and self._online_cache_mismatches == 0
        )
        return obs, reward, terminated, truncated, info

    # ==================================================================
    #   Filter / State estimation
    # ==================================================================

    def _prepare_filter_cache(self):
        """Precompute KF estimates with burst mismatch for each obstacle."""
        self.dyn_hat_world, self.dyn_tk, self.dyn_P_world = [], [], []
        self._kf_cache_hit = False
        self._kf_cache_path = None
        if self.dyn_traj is None or self.dyn_obs_num <= 0:
            return
        if self._load_filter_cache():
            if self.filter_execution_mode == "online_exact":
                self._kf_reference_hat = np.asarray(
                    self.dyn_hat_world, dtype=np.float32
                )
                self._kf_reference_tk = np.asarray(
                    self.dyn_tk, dtype=np.float32
                )
                self._kf_reference_p = np.asarray(
                    self.dyn_P_world, dtype=np.float32
                )
                self._initialize_online_filter_tracks()
            return
        T = int(self.dyn_traj.shape[0])
        for i in range(self.dyn_obs_num):
            gt = self.dyn_traj[:, i, :4].astype(float)
            obs_seed = int(self.dyn_seeds[i]) if self.dyn_seeds is not None and i < len(self.dyn_seeds) else int(self.rng.integers(0, 2**31 - 1))
            x_hat, tk, P_hist = self._run_filter_with_burst(gt, obs_seed, obs_id=i)
            self.dyn_hat_world.append(x_hat)
            self.dyn_tk.append(tk)
            self.dyn_P_world.append(P_hist)
        self._write_filter_cache()

    def _kf_cache_read_enabled(self) -> bool:
        return bool(self.kf_cache_dirs) and self.kf_cache_mode in ("read", "read_write", "read_strict")

    def _kf_cache_write_enabled(self) -> bool:
        return self.kf_cache_dir is not None and self.kf_cache_mode in ("write", "read_write")

    def _kf_cache_strict_read_enabled(self) -> bool:
        return bool(self.kf_cache_dirs) and self.kf_cache_mode == "read_strict"

    def _scenario_digest(self) -> str:
        h = hashlib.sha256()
        for value in (
            self.dyn_traj,
            self.dyn_seeds,
            self.fixed_initial_position,
            self.fixed_goal,
        ):
            if value is None:
                h.update(b"<none>")
                continue
            arr = np.ascontiguousarray(value)
            h.update(str(arr.shape).encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(arr.view(np.uint8))
        return h.hexdigest()

    def _kf_cache_metadata(self) -> dict:
        ni = self.noise_injector
        metadata = {
            "cache_version": 1,
            "scenario_digest": self._scenario_digest(),
            "scenario_T": int(self.scenario_T),
            "dyn_obs_num": int(self.dyn_obs_num),
            "dt": float(self.dt),
            "max_episode_steps": int(self.max_episode_steps),
            "start_mode": str(ni.start_mode),
            "burst_start_step": int(getattr(ni, "_burst_start_step", -1)),
            "burst_duration_steps": int(ni.burst_duration_steps),
            "measurement_cov_scale": float(ni.measurement_cov_scale),
            "measurement_delay_steps": int(ni.measurement_delay_steps),
            "bias_enabled": bool(ni.bias_enabled),
            "bias_position_y": float(ni.bias_position_y),
            "bias_duration_steps": int(ni.bias_duration_steps),
            "nominal_position_std": float(ni.nominal_position_std),
            "nominal_velocity_std": float(ni.nominal_velocity_std),
            "path_progress_arrival_steps": int(self.path_progress_arrival_steps),
            "risk_distance_threshold": float(ni.risk_distance_threshold),
            "tcpa_horizon": float(ni.tcpa_horizon),
            "trust_mode": str(self.trust_mode),
            "trust_t_min": float(self.trust_computer.t_min),
            "trust_t_max": float(self.trust_computer.t_max),
        }
        if str(ni.start_mode) in (
            "path_progress_multi_binned_candidate",
            "path_progress_topk_binned_candidate",
        ):
            metadata["burst_start_steps"] = [
                int(x) for x in getattr(ni, "_burst_start_steps", [])
            ]
            metadata["path_progress_candidate_stride"] = int(self.path_progress_candidate_stride)
        if str(ni.start_mode) == "path_progress_multi_binned_candidate":
            metadata["path_progress_max_pulses"] = int(self.path_progress_max_pulses)
        if str(ni.start_mode) == "path_progress_topk_binned_candidate":
            metadata["path_progress_topk"] = int(self.path_progress_topk)
            metadata["path_progress_min_separation_steps"] = int(self.path_progress_min_separation_steps)
            metadata["path_progress_anchor_lead_steps"] = int(self.path_progress_anchor_lead_steps)
            metadata["path_progress_candidate_distance_threshold"] = (
                None
                if self.path_progress_candidate_distance_threshold is None
                else float(self.path_progress_candidate_distance_threshold)
            )
        return metadata

    def _resolve_kf_cache_paths(self) -> tuple[list[pathlib.Path], dict]:
        metadata = self._kf_cache_metadata()
        text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if self._current_scenario_path is not None:
            stem = self._current_scenario_path.stem
        else:
            stem = f"manual_{metadata['scenario_digest'][:12]}"
        return [root / fingerprint / f"{stem}.npz" for root in self.kf_cache_dirs], metadata

    def _resolve_kf_cache_path(self) -> tuple[pathlib.Path, dict]:
        paths, metadata = self._resolve_kf_cache_paths()
        if not paths:
            raise RuntimeError("KF replay cache path requested without kf_cache_dir")
        return paths[0], metadata

    def _load_filter_cache(self) -> bool:
        if not self._kf_cache_read_enabled():
            return False
        paths, metadata = self._resolve_kf_cache_paths()
        self._kf_cache_path = paths[0] if paths else None
        strict = self._kf_cache_strict_read_enabled()
        last_miss = f"missing cache entry {', '.join(str(p) for p in paths)}"
        for path in paths:
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    saved_metadata = json.loads(str(data["metadata_json"]))
                    if saved_metadata != metadata:
                        last_miss = f"metadata mismatch for {path}"
                        continue
                    dyn_hat = data["dyn_hat_world"]
                    dyn_tk = data["dyn_tk"]
                    dyn_p = data["dyn_P_world"]
            except Exception as exc:
                last_miss = f"failed to load {path}: {exc}"
                continue
            if dyn_hat.shape[0] < self.dyn_obs_num or dyn_tk.shape[0] < self.dyn_obs_num or dyn_p.shape[0] < self.dyn_obs_num:
                last_miss = f"shape mismatch for {path}"
                continue
            self._kf_cache_path = path
            self.dyn_hat_world = [dyn_hat[i].copy() for i in range(self.dyn_obs_num)]
            self.dyn_tk = [dyn_tk[i].copy() for i in range(self.dyn_obs_num)]
            self.dyn_P_world = [dyn_p[i].copy() for i in range(self.dyn_obs_num)]
            self._kf_cache_hit = True
            return True
        if strict:
            raise RuntimeError(f"Strict KF replay cache miss: {last_miss}")
        return False

    def _write_filter_cache(self) -> None:
        if not self._kf_cache_write_enabled() or not self.dyn_hat_world:
            return
        path, metadata = self._resolve_kf_cache_path()
        self._kf_cache_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{id(self)}.tmp")
        with tmp_path.open("wb") as f:
            np.savez(
                f,
                dyn_hat_world=np.asarray(self.dyn_hat_world, dtype=np.float32),
                dyn_tk=np.asarray(self.dyn_tk, dtype=np.float32),
                dyn_P_world=np.asarray(self.dyn_P_world, dtype=np.float32),
                metadata_json=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            )
        try:
            tmp_path.replace(path)
        except PermissionError:
            # Windows can deny replacement when another worker writes or reads
            # the same cache key. The freshly computed KF replay remains valid;
            # cache persistence is only a speed optimization.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _path_progress_ship_xy(self, step: int) -> np.ndarray:
        """Approximate ownship position along the start-goal path."""
        start = getattr(self, "fixed_initial_position", None)
        if start is None:
            start = getattr(self, "initial_position", self.ship_state[:2])
        start = np.asarray(start, dtype=float)[:2]
        goal = np.asarray(getattr(self, "goal", start), dtype=float)[:2]
        denom = max(1.0, float(getattr(self, "path_progress_arrival_steps", 230)))
        alpha = float(np.clip(float(step) / denom, 0.0, 1.0))
        return start + alpha * (goal - start)

    def _configure_path_progress_burst_from_replay(self) -> None:
        """Set one global path-progress burst start from replayed obstacle tracks."""
        if self.dyn_traj is None or self.dyn_obs_num <= 0:
            return
        mode = getattr(self.noise_injector, "start_mode", None)
        candidate_modes = (
            "path_progress_random_candidate",
            "path_progress_binned_candidate",
            "path_progress_multi_binned_candidate",
            "path_progress_topk_binned_candidate",
        )
        if mode not in ("path_progress", *candidate_modes):
            return
        if not getattr(self.noise_injector, "_burst_allowed", False):
            return

        threshold = float(self.noise_injector.risk_distance_threshold)
        candidate_distance_threshold = getattr(self, "path_progress_candidate_distance_threshold", None)
        candidate_threshold = (
            threshold
            if candidate_distance_threshold is None
            else float(candidate_distance_threshold)
        )
        T = int(self.dyn_traj.shape[0])
        effective_horizon = int(getattr(self.noise_injector, "_effective_horizon", T))
        if mode == "path_progress_topk_binned_candidate":
            starts = select_topk_binned_candidate_steps(
                dyn_traj=self.dyn_traj[:, : self.dyn_obs_num, :],
                start_xy=getattr(self, "fixed_initial_position", self.initial_position),
                goal_xy=self.goal,
                arrival_steps=self.path_progress_arrival_steps,
                effective_horizon=effective_horizon,
                distance_threshold=candidate_threshold,
                stride=self.path_progress_candidate_stride,
                top_k=self.path_progress_topk,
                min_separation_steps=self.path_progress_min_separation_steps,
                anchor_lead_steps=getattr(self, "path_progress_anchor_lead_steps", 0),
            )
            if starts.size:
                self.noise_injector.set_burst_start_steps([int(x) for x in starts.tolist()])
                self.noise_injector._burst_mode_used = mode
            else:
                self.noise_injector._burst_armed = False
            return

        candidates = compute_path_progress_candidate_steps(
            dyn_traj=self.dyn_traj[:, : self.dyn_obs_num, :],
            start_xy=getattr(self, "fixed_initial_position", self.initial_position),
            goal_xy=self.goal,
            arrival_steps=self.path_progress_arrival_steps,
            risk_distance_threshold=candidate_threshold,
            effective_horizon=effective_horizon,
        )
        if mode == "path_progress":
            if candidates:
                self.noise_injector._burst_start_step = int(candidates[0])
                self.noise_injector._burst_armed = False
                self.noise_injector._burst_mode_used = mode
            return

        if not candidates:
            self.noise_injector._burst_armed = False
            return

        forced = getattr(self, "forced_burst_start_step", None)
        if forced is not None:
            forced = int(forced)
            if forced not in set(candidates):
                raise ValueError(f"forced_burst_start_step={forced} is not a valid path-progress candidate")
            self.noise_injector._burst_start_step = forced
            self.noise_injector._burst_armed = False
            self.noise_injector._burst_mode_used = mode
            return

        if mode == "path_progress_random_candidate":
            rng = getattr(self, "rng", np.random.default_rng())
            idx = int(rng.integers(0, len(candidates)))
            self.noise_injector._burst_start_step = int(candidates[idx])
            self.noise_injector._burst_armed = False
            self.noise_injector._burst_mode_used = mode
            return

        if mode == "path_progress_binned_candidate":
            rng = getattr(self, "rng", np.random.default_rng())
            self.noise_injector._burst_start_step = choose_raw_then_binned_candidate(
                candidates,
                stride=self.path_progress_candidate_stride,
                rng=rng,
            )
            self.noise_injector._burst_armed = False
            self.noise_injector._burst_mode_used = mode
            return

        if mode == "path_progress_multi_binned_candidate":
            starts = select_multi_binned_candidate_steps(
                candidates,
                stride=self.path_progress_candidate_stride,
                max_pulses=self.path_progress_max_pulses,
            )
            self.noise_injector.set_burst_start_steps([int(x) for x in starts.tolist()])
            self.noise_injector._burst_mode_used = mode

    def _filter_model(self) -> tuple[np.ndarray, ...]:
        dt = self.dt
        f = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        h = np.eye(4, dtype=float)
        identity = np.eye(4, dtype=float)
        q_nom = self._cv_process_Q(0.35, dt)
        pos_std = self.noise_injector.nominal_position_std
        vel_std = self.noise_injector.nominal_velocity_std
        r_nom = np.diag([
            pos_std**2,
            pos_std**2,
            vel_std**2,
            vel_std**2,
        ]).astype(float)
        return f, h, identity, q_nom, r_nom

    def _sample_filter_measurement(
        self,
        *,
        gt: np.ndarray,
        k: int,
        rng: np.random.Generator,
        previous_estimate: np.ndarray,
    ) -> np.ndarray:
        if str(self.noise_injector.start_mode).startswith("path_progress"):
            ship_xy = self._path_progress_ship_xy(k)
            dx = float(previous_estimate[0] - ship_xy[0])
            dy = float(previous_estimate[1] - ship_xy[1])
        else:
            dx = (
                float(previous_estimate[0] - self.ship_state[0])
                if hasattr(self, "ship_state") else 10.0
            )
            dy = (
                float(previous_estimate[1] - self.ship_state[1])
                if hasattr(self, "ship_state") else 10.0
            )
        risk_active = math.hypot(dx, dy) < (
            self.noise_injector.risk_distance_threshold
        )
        r_true = self.noise_injector.get_true_measurement_covariance(
            k, risk_active
        )
        r_true = 0.5 * (r_true + r_true.T) + 1e-12 * np.eye(4)
        bias = self.noise_injector.get_bias(k)
        delay = self.noise_injector.measurement_delay_steps
        burst_now = self.noise_injector.is_burst(k, risk_active)
        source_k = max(0, k - delay) if delay > 0 and burst_now else k
        measurement = gt[source_k] + rng.multivariate_normal(
            np.zeros(4), r_true
        )
        measurement[1] += bias
        return measurement

    def _filter_step(
        self,
        *,
        gt: np.ndarray,
        k: int,
        obs_id: int,
        rng: np.random.Generator,
        x_prev: np.ndarray,
        p_prev: np.ndarray,
        model: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, int]]:
        f, h, identity, q_nom, r_nom = model
        measurement_start = time.perf_counter_ns()
        z = self._sample_filter_measurement(
            gt=gt,
            k=k,
            rng=rng,
            previous_estimate=x_prev,
        )
        measurement_ns = time.perf_counter_ns() - measurement_start

        filter_start = time.perf_counter_ns()
        x_pred = f @ x_prev
        p_pred = f @ p_prev @ f.T + q_nom
        innovation = z - h @ x_pred
        s = h @ p_pred @ h.T + r_nom
        s = 0.5 * (s + s.T) + 1e-9 * identity
        gain = p_pred @ h.T @ np.linalg.inv(s)
        x_upd = x_pred + gain @ innovation
        p_upd = (
            (identity - gain @ h) @ p_pred @ (identity - gain @ h).T
            + gain @ r_nom @ gain.T
        )
        p_upd = 0.5 * (p_upd + p_upd.T)
        filter_ns = time.perf_counter_ns() - filter_start

        trust_start = time.perf_counter_ns()
        if self.trust_mode == "oracle_tmse":
            trust = self.trust_computer.compute_per_target(
                x_true=gt[k], x_hat=x_upd, P_f=p_upd,
            )
        else:
            nu, s_k = self.trust_computer.compute_innovations_batch(
                z, x_pred, h, p_pred, r_nom,
            )
            trust = self.trust_computer.update(
                obs_id=obs_id, innovation=nu, S_k=s_k,
            )
        trust = float(np.clip(
            trust,
            self.trust_computer.t_min,
            self.trust_computer.t_max,
        ))
        trust_ns = time.perf_counter_ns() - trust_start
        return x_upd, p_upd, trust, {
            "measurement_sim_ns": measurement_ns,
            "filter_ns": filter_ns,
            "trust_ns": trust_ns,
        }

    def _run_filter_statefully(
        self,
        gt: np.ndarray,
        seed: int,
        obs_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        track = np.asarray(gt, dtype=float)
        x_hist = np.zeros((track.shape[0], 4), dtype=float)
        p_hist = np.zeros((track.shape[0], 4, 4), dtype=float)
        trust_hist = np.ones(track.shape[0], dtype=float)
        x_hist[0] = track[0] + rng.normal(
            0.0, [0.10, 0.10, 0.10, 0.10], size=4
        )
        p_hist[0] = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)
        model = self._filter_model()
        for k in range(1, track.shape[0]):
            x_hist[k], p_hist[k], trust_hist[k], _ = self._filter_step(
                gt=track,
                k=k,
                obs_id=obs_id,
                rng=rng,
                x_prev=x_hist[k - 1],
                p_prev=p_hist[k - 1],
                model=model,
            )
        return x_hist, trust_hist, p_hist

    def _initialize_online_filter_tracks(self) -> None:
        if self.dyn_seeds is None or len(self.dyn_seeds) < self.dyn_obs_num:
            raise RuntimeError("online_exact requires one dyn_seed per obstacle")
        self._online_filter_model = self._filter_model()
        self._online_filter_rngs = []
        self._online_filter_x = []
        self._online_filter_p = []
        self._online_filter_gt = []
        self.dyn_hat_world = [[] for _ in range(self.dyn_obs_num)]
        self.dyn_tk = [[] for _ in range(self.dyn_obs_num)]
        self.dyn_P_world = [[] for _ in range(self.dyn_obs_num)]
        for obstacle_id in range(self.dyn_obs_num):
            track = self.dyn_traj[:, obstacle_id, :4].astype(float)
            rng = np.random.default_rng(int(self.dyn_seeds[obstacle_id]))
            x0 = track[0] + rng.normal(
                0.0, [0.10, 0.10, 0.10, 0.10], size=4
            )
            p0 = np.eye(4, dtype=float)
            self._online_filter_gt.append(track)
            self._online_filter_rngs.append(rng)
            self._online_filter_x.append(x0)
            self._online_filter_p.append(p0)
            self.dyn_hat_world[obstacle_id].append(x0.astype(np.float32))
            self.dyn_tk[obstacle_id].append(np.float32(1.0))
            self.dyn_P_world[obstacle_id].append(p0.astype(np.float32))
        self._last_online_filter_timing_ns = {
            "estimator_step": 0,
            "measurement_sim_ns": 0,
            "filter_ns": 0,
            "trust_ns": 0,
        }

    def _advance_online_filter_to(self, step: int) -> None:
        if self.filter_execution_mode != "online_exact":
            return
        if self._online_filter_model is None:
            raise RuntimeError("online filter tracks are not initialized")
        target = int(step)
        if target < 0 or target >= self.scenario_T:
            raise IndexError(f"online filter step {target} is out of range")
        while len(self.dyn_hat_world[0]) <= target:
            k = len(self.dyn_hat_world[0])
            measurement_ns = 0
            filter_ns = 0
            trust_ns = 0
            for obstacle_id in range(self.dyn_obs_num):
                x_upd, p_upd, trust, timing = self._filter_step(
                    gt=self._online_filter_gt[obstacle_id],
                    k=k,
                    obs_id=obstacle_id,
                    rng=self._online_filter_rngs[obstacle_id],
                    x_prev=self._online_filter_x[obstacle_id],
                    p_prev=self._online_filter_p[obstacle_id],
                    model=self._online_filter_model,
                )
                measurement_ns += timing["measurement_sim_ns"]
                filter_ns += timing["filter_ns"]
                trust_ns += timing["trust_ns"]
                snapshot_started = time.perf_counter_ns()
                self._online_filter_x[obstacle_id] = x_upd
                self._online_filter_p[obstacle_id] = p_upd
                self.dyn_hat_world[obstacle_id].append(
                    x_upd.astype(np.float32)
                )
                self.dyn_tk[obstacle_id].append(np.float32(trust))
                self.dyn_P_world[obstacle_id].append(
                    p_upd.astype(np.float32)
                )
                filter_ns += time.perf_counter_ns() - snapshot_started
            self._last_online_filter_timing_ns = {
                "estimator_step": k,
                "measurement_sim_ns": int(measurement_ns),
                "filter_ns": int(filter_ns),
                "trust_ns": int(trust_ns),
            }

    def _require_cache_equal(
        self,
        field: str,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        step: int,
        obstacle_id: int,
    ) -> None:
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        if np.array_equal(actual_array, expected_array):
            return
        self._online_cache_mismatches += 1
        map_name = (
            self._current_scenario_path.name
            if self._current_scenario_path is not None
            else "<manual>"
        )
        if actual_array.shape != expected_array.shape:
            detail = (
                f"actual_shape={actual_array.shape} "
                f"expected_shape={expected_array.shape}"
            )
        else:
            mismatch = np.argwhere(actual_array != expected_array)[0].tolist()
            index = tuple(mismatch)
            detail = (
                f"index={mismatch} actual={actual_array[index]} "
                f"expected={expected_array[index]}"
            )
        raise OnlineFilterCacheMismatch(
            f"map={map_name} seed={self._last_seed} "
            f"anchor={getattr(self.noise_injector, '_burst_start_step', -1)} "
            f"step={step} obstacle={obstacle_id} field={field} {detail}"
        )

    def _cache_reference_obstacle_estimates(self, step: int) -> dict[int, dict]:
        if self._kf_reference_hat is None or self._kf_reference_tk is None:
            raise RuntimeError("online cache reference is not initialized")
        usv_state = self.ship_state
        psi_usv = float(usv_state[2])
        c, s = np.cos(-psi_usv), np.sin(-psi_usv)
        estimates = {}
        for obstacle_id in range(self.dyn_obs_num):
            xw, yw, vx_w, vy_w = map(
                float, self._kf_reference_hat[obstacle_id, step]
            )
            trust = float(self._kf_reference_tk[obstacle_id, step])
            lx, ly = world_to_local(xw, yw, usv_state)
            estimates[obstacle_id] = {
                "dx": float(lx),
                "dy": float(ly),
                "vx": float(c * vx_w - s * vy_w),
                "vy": float(s * vx_w + c * vy_w),
                "radius": float(self.dyn_radius),
                "conf": float(np.clip(trust, 0.0, 1.0)),
                "trust": float(np.clip(
                    trust,
                    self.trust_computer.t_min,
                    self.trust_computer.t_max,
                )),
            }
        return estimates

    def _assert_online_cache_step(self, step: int) -> None:
        if self.filter_execution_mode != "online_exact":
            return
        if (
            self._kf_reference_hat is None
            or self._kf_reference_tk is None
            or self._kf_reference_p is None
        ):
            raise RuntimeError("online cache reference is not initialized")
        k = int(step)
        for obstacle_id in range(self.dyn_obs_num):
            self._require_cache_equal(
                "dyn_hat_world",
                self.dyn_hat_world[obstacle_id][k],
                self._kf_reference_hat[obstacle_id, k],
                step=k,
                obstacle_id=obstacle_id,
            )
            self._require_cache_equal(
                "dyn_tk",
                self.dyn_tk[obstacle_id][k],
                self._kf_reference_tk[obstacle_id, k],
                step=k,
                obstacle_id=obstacle_id,
            )
            self._require_cache_equal(
                "dyn_P_world",
                self.dyn_P_world[obstacle_id][k],
                self._kf_reference_p[obstacle_id, k],
                step=k,
                obstacle_id=obstacle_id,
            )
            self._online_cache_checks += 1

        actual_observation = self._get_obs()
        actual_policy_ids = tuple(self._last_policy_obstacle_ids)
        actual_t_min = self._last_t_min
        actual_estimates = self.obstacle_estimates
        expected_estimates = self._cache_reference_obstacle_estimates(k)
        self._require_cache_equal(
            "active_obstacle_ids",
            np.asarray(sorted(actual_estimates), dtype=np.int64),
            np.asarray(sorted(expected_estimates), dtype=np.int64),
            step=k,
            obstacle_id=-1,
        )
        try:
            self.obstacle_estimates = expected_estimates
            expected_observation = self._get_obs()
            expected_policy_ids = tuple(self._last_policy_obstacle_ids)
        finally:
            self.obstacle_estimates = actual_estimates
            self._last_policy_obstacle_ids = actual_policy_ids
            self._last_t_min = actual_t_min
        self._require_cache_equal(
            "policy_obstacle_ids",
            np.asarray(actual_policy_ids, dtype=np.int64),
            np.asarray(expected_policy_ids, dtype=np.int64),
            step=k,
            obstacle_id=-1,
        )
        for key in actual_observation:
            self._require_cache_equal(
                f"observation.{key}",
                actual_observation[key],
                expected_observation[key],
                step=k,
                obstacle_id=-1,
            )

    def _run_filter_with_burst(self, gt: np.ndarray, seed: int, obs_id: int = 0):
        """Run KF with burst mismatch injection for one obstacle.

        Returns: (x_hat, tk, P_hist)
        """
        return self._run_filter_statefully(gt, seed, obs_id=obs_id)

    @staticmethod
    def _cv_process_Q(sigma_a: float, dt: float) -> np.ndarray:
        q = sigma_a ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q1 = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]], dtype=float) * q
        Q = np.zeros((4, 4), dtype=float)
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        return Q

    def _filter_dynamic_obstacles(self):
        """Use precomputed filter trajectories to fill obstacle_estimates."""
        if not hasattr(self, "obstacle_estimates"):
            self.obstacle_estimates = {}
        if not hasattr(self, "dyn_hat_world") or len(self.dyn_hat_world) < self.dyn_obs_num:
            self._prepare_filter_cache()

        usv_state = self.ship_state
        psi_usv = float(usv_state[2])
        c, s = np.cos(-psi_usv), np.sin(-psi_usv)
        k = int(self.dyn_step)
        self._advance_online_filter_to(k)

        self.obstacle_estimates.clear()
        coordinate_ns = 0
        for i in range(self.dyn_obs_num):
            # ASSERT: k must be within precomputed filter cache range.
            # No ground-truth fallback is allowed.
            if i >= len(self.dyn_hat_world) or k >= len(self.dyn_hat_world[i]):
                self._ground_truth_fallback_used = True
                raise RuntimeError(
                    f"Ground-truth fallback triggered! "
                    f"dyn_step={k}, cache_len={len(self.dyn_hat_world[i]) if i < len(self.dyn_hat_world) else 'N/A'}, "
                    f"scenario_T={self.scenario_T}. Episode should have been truncated."
                )

            coordinate_started = time.perf_counter_ns()
            xw, yw, vx_w, vy_w = map(float, self.dyn_hat_world[i][k])
            conf = float(self.dyn_tk[i][k])
            trust_val = float(self.dyn_tk[i][k])

            lx, ly = world_to_local(xw, yw, usv_state)
            vx_l = c * vx_w - s * vy_w
            vy_l = s * vx_w + c * vy_w

            self.obstacle_estimates[i] = dict(
                dx=float(lx), dy=float(ly),
                vx=float(vx_l), vy=float(vy_l),
                radius=float(self.dyn_radius),
                conf=float(np.clip(conf, 0.0, 1.0)),
                trust=float(np.clip(trust_val,
                                    self.trust_computer.t_min,
                                    self.trust_computer.t_max)),
            )
            coordinate_ns += time.perf_counter_ns() - coordinate_started
            self.dyn_est_traj[i].append(np.array([xw, yw], dtype=float))
        self._last_coordinate_ns = int(coordinate_ns)

    def _use_ground_truth_obstacles(self):
        """Fill estimates with perfect ground truth (use_filter=False)."""
        self._last_coordinate_ns = 0
        if not hasattr(self, "obstacle_estimates"):
            self.obstacle_estimates = {}
        usv_state = self.ship_state
        psi_usv = float(usv_state[2])
        for i in range(self.dyn_obs_num):
            xw, yw, vx_w, vy_w = map(float, self.dyn_pos[i])
            lx, ly = world_to_local(xw, yw, usv_state)
            c, s = np.cos(-psi_usv), np.sin(-psi_usv)
            vx_l = c * vx_w - s * vy_w
            vy_l = s * vx_w + c * vy_w
            self.obstacle_estimates[i] = dict(
                dx=lx, dy=ly, vx=vx_l, vy=vy_l,
                radius=self.dyn_radius, conf=1.0, trust=1.0,
            )
            self.dyn_est_traj[i].append(np.array([xw, yw], dtype=float))

    def _compute_min_dcpa(self) -> float:
        """Compute minimum DCPA across all obstacles (body frame)."""
        if self.dyn_obs_num == 0:
            return float("inf")
        u_own = float(self.ship_state[3])
        v_own = float(self.ship_state[4])
        min_d = float("inf")
        for est in self.obstacle_estimates.values():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            vx_obs = float(est.get("vx", 0.0))
            vy_obs = float(est.get("vy", 0.0))
            vx_rel = vx_obs - u_own
            vy_rel = vy_obs - v_own
            v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
            if v_rel_sq < 1e-6:
                d = float(np.hypot(dx, dy))
            else:
                tcpa = -(dx * vx_rel + dy * vy_rel) / v_rel_sq
                if tcpa <= 0:
                    d = float(np.hypot(dx, dy))
                else:
                    cx = dx + vx_rel * tcpa
                    cy = dy + vy_rel * tcpa
                    d = float(np.hypot(cx, cy))
            if d < min_d:
                min_d = d
        return min_d

    def _check_any_risk_active(self) -> bool:
        """Check if any obstacle is risk-active (for diagnostics)."""
        u_own = float(self.ship_state[3])
        v_own = float(self.ship_state[4])
        for est in self.obstacle_estimates.values():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            dist = math.hypot(dx, dy)
            if dist >= self.noise_injector.risk_distance_threshold:
                continue
            vx_obs = float(est.get("vx", 0.0))
            vy_obs = float(est.get("vy", 0.0))
            vx_rel = vx_obs - u_own
            vy_rel = vy_obs - v_own
            v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
            if v_rel_sq > 1e-6:
                tcpa = -(dx * vx_rel + dy * vy_rel) / v_rel_sq
                if 0.0 < tcpa < self.noise_injector.tcpa_horizon:
                    return True
        return False

    # ==================================================================
    #   Optional CI-VO shield
    # ==================================================================

    @staticmethod
    def _sigmoid_clip(x: float) -> float:
        return float(1.0 / (1.0 + math.exp(-float(np.clip(x, -60.0, 60.0)))))

    def _colregs_relative_bearing(self, dx: float, dy: float) -> float:
        return float(math.atan2(float(dy), float(dx)))

    def _colregs_memberships(self, beta: float) -> tuple[float, float]:
        theta_h = float(self.rc_colregs_theta_h)
        theta_lim = float(self.rc_colregs_theta_lim)
        kappa = float(self.rc_colregs_kappa_beta)
        psi_head = self._sigmoid_clip(kappa * (math.cos(beta) - math.cos(theta_h)))
        psi_crossing = self._sigmoid_clip(
            kappa * (-beta - theta_h)
        ) * self._sigmoid_clip(kappa * (beta + theta_lim))
        return float(psi_head), float(psi_crossing)

    def _colregs_tcpa_dcpa(self, dx: float, dy: float, vx_obs: float, vy_obs: float) -> tuple[float, float]:
        rx = float(vx_obs) - float(self.ship_state[3])
        ry = float(vy_obs) - float(self.ship_state[4])
        vel2 = rx * rx + ry * ry
        if vel2 < 1e-9:
            return -1.0, float(math.hypot(dx, dy))
        tcpa = -((float(dx) * rx + float(dy) * ry) / vel2)
        if tcpa <= 0.0:
            return float(tcpa), float(math.hypot(dx, dy))
        dcpa = math.hypot(float(dx) + rx * tcpa, float(dy) + ry * tcpa)
        return float(tcpa), float(dcpa)

    def _colregs_risk_gate(self, dx: float, dy: float, vx_obs: float, vy_obs: float) -> float:
        tcpa, dcpa = self._colregs_tcpa_dcpa(dx, dy, vx_obs, vy_obs)
        if not (np.isfinite(tcpa) and np.isfinite(dcpa)) or tcpa <= 0.0:
            return 0.0
        spatial = max(0.0, 1.0 - float(dcpa) / self.rc_colregs_d_safe)
        temporal = self._sigmoid_clip(self.rc_colregs_kappa_time * tcpa) * self._sigmoid_clip(
            self.rc_colregs_kappa_time * (self.rc_colregs_tau - tcpa)
        )
        return float(np.clip(spatial * temporal, 0.0, 1.0))

    def _colregs_phi_single(self, dx: float, dy: float, vx_obs: float, vy_obs: float) -> float:
        signal = colregs_duty(
            relative_position=np.array([dx, dy], dtype=float),
            own_velocity=np.asarray(self.ship_state[3:5], dtype=float),
            target_velocity=np.array([vx_obs, vy_obs], dtype=float),
            d_safe=self.rc_colregs_d_safe,
            tcpa_horizon=self.rc_colregs_tau,
            kappa_beta=self.rc_colregs_kappa_beta,
            kappa_time=self.rc_colregs_kappa_time,
            theta_head=self.rc_colregs_theta_h,
            theta_limit=self.rc_colregs_theta_lim,
        )
        return float(signal["phi"])

    def _update_colregs_signal(self) -> None:
        self.colregs_phi_by_id = {}
        self.colregs_beta_by_id = {}
        self._last_colregs_aggregate_duty = 0.0
        if not self.rc_colregs_enabled or not self.obstacle_estimates:
            return
        for obs_id, est in self.obstacle_estimates.items():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            self.colregs_phi_by_id[int(obs_id)] = self._colregs_phi_single(
                dx,
                dy,
                float(est.get("vx", 0.0)),
                float(est.get("vy", 0.0)),
            )
            self.colregs_beta_by_id[int(obs_id)] = self._colregs_relative_bearing(dx, dy)
        self._last_colregs_aggregate_duty = aggregate_colregs_duty(
            self.colregs_phi_by_id.values()
        )

    def _colregs_turn_violation(self, raw_yaw: float) -> float:
        starboard_cmd = self.rc_colregs_starboard_yaw_sign * float(np.clip(raw_yaw, -1.0, 1.0))
        return max(0.0, self.rc_colregs_required_starboard - starboard_cmd)

    def _collect_colregs_metrics(self, raw_yaw: float) -> dict[str, float | int | bool]:
        self._update_colregs_signal()
        active = [v for v in self.colregs_phi_by_id.values() if float(v) > 0.0]
        required = [v for v in active if float(v) >= 0.05]
        violation = self._colregs_turn_violation(raw_yaw) if required else 0.0
        compliant = len(required) if violation <= 1e-9 else 0
        return {
            "rc_colregs_enabled": bool(self.rc_colregs_enabled),
            "colregs_active_count": int(len(active)),
            "colregs_required_count": int(len(required)),
            "colregs_compliant_count": int(compliant),
            "colregs_phi_sum": float(np.sum(active)) if active else 0.0,
            "colregs_phi_bar": float(self._last_colregs_aggregate_duty),
            "colregs_violation": float(violation * len(required)),
            "colregs_compliance": float(compliant / len(required) if required else 1.0),
        }

    def _civo_active(self) -> bool:
        return bool(self.civo_enabled or self.civo_shield_enabled)

    def _credible_covariance_for_obstacle(self, obs_id: int) -> np.ndarray:
        k = int(self.dyn_step)
        try:
            true_state = np.asarray(self.dyn_traj[k, obs_id, :4], dtype=float)
            estimated_state = np.asarray(self.dyn_hat_world[obs_id][k], dtype=float).reshape(4)
            filter_covariance = np.asarray(self.dyn_P_world[obs_id][k], dtype=float).reshape(4, 4)
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"credible CoReCBF requires Pf, oracle state, and posterior estimate for obstacle {obs_id}"
            ) from exc
        error = true_state - estimated_state
        return credible_covariance(filter_covariance, np.outer(error, error))

    def _civo_projection_terms(
        self,
        obs_id: int,
        dx: float,
        dy: float,
        ship_psi: Optional[float] = None,
    ) -> Optional[tuple[float, float]]:
        """Project filter position covariance onto LOS and lateral axes."""
        if not hasattr(self, "dyn_P_world") or obs_id >= len(self.dyn_P_world):
            return None
        k = int(self.dyn_step)
        if k >= len(self.dyn_P_world[obs_id]):
            return None

        if self.corecbf_variant == "credibility_colregs":
            P_world = self._credible_covariance_for_obstacle(obs_id)[:2, :2]
        else:
            P_world = np.asarray(self.dyn_P_world[obs_id][k], dtype=float).reshape(4, 4)[:2, :2]
        psi = float(self.ship_state[2] if ship_psi is None else ship_psi)
        c, s = np.cos(-psi), np.sin(-psi)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        P_body = rot @ P_world @ rot.T
        P_body = 0.5 * (P_body + P_body.T)

        r_vec = np.array([float(dx), float(dy)], dtype=float)
        dist = float(np.linalg.norm(r_vec))
        if dist < 1e-9:
            return None

        r_hat = r_vec / dist
        r_perp = np.array([-r_hat[1], r_hat[0]], dtype=float)
        e_parallel = self._civo_sigma_scale * math.sqrt(max(float(r_hat @ P_body @ r_hat), 0.0))
        e_perp = self._civo_sigma_scale * math.sqrt(max(float(r_perp @ P_body @ r_perp), 0.0))
        return float(e_parallel), float(e_perp)

    def _civo_half_angle(
        self,
        obs_id: int,
        dx: float,
        dy: float,
        r_sum: float,
        base_alpha: float,
        ship_psi: Optional[float] = None,
    ) -> tuple[float, dict[str, float]]:
        terms = self._civo_projection_terms(obs_id, dx, dy, ship_psi=ship_psi)
        dist = float(math.hypot(dx, dy))
        if terms is None or dist < 1e-9:
            return float(base_alpha), {
                "civo_e_parallel": 0.0,
                "civo_e_perp": 0.0,
                "civo_d_min": dist,
            }

        e_parallel, e_perp = terms
        d_min = dist - e_parallel
        if d_min <= float(r_sum):
            return math.pi, {
                "civo_e_parallel": float(e_parallel),
                "civo_e_perp": float(e_perp),
                "civo_d_min": float(d_min),
            }

        delta_pos = math.atan2(float(e_perp), float(d_min))
        beta_rad = math.asin(np.clip(float(r_sum) / float(d_min), -1.0, 1.0))
        alpha = min(math.pi, delta_pos + beta_rad)
        return float(alpha), {
            "civo_e_parallel": float(e_parallel),
            "civo_e_perp": float(e_perp),
            "civo_d_min": float(d_min),
            "civo_delta_pos": float(delta_pos),
            "civo_beta_rad": float(beta_rad),
        }

    def _compute_civo_single(
        self,
        obs_id: int,
        est: Dict,
        ship_psi: Optional[float] = None,
    ) -> Optional[dict]:
        dx = float(est.get("dx", 0.0))
        dy = float(est.get("dy", 0.0))
        dist = float(math.hypot(dx, dy))
        if dist < 1e-6:
            return None

        r_sum = float(self.R_usv + float(est.get("radius", self.dyn_radius)))
        if self.civo_shield_enabled:
            r_sum = max(r_sum, self.corecbf_safety_distance)
        angle = math.atan2(dy, dx)
        base_alpha = 0.5 * math.pi if dist <= r_sum else math.asin(np.clip(r_sum / dist, -1.0, 1.0))
        alpha = float(base_alpha)
        extra: dict[str, float] = {}
        if self._civo_active():
            alpha, extra = self._civo_half_angle(obs_id, dx, dy, r_sum, base_alpha, ship_psi=ship_psi)
        civo_delta = max(0.0, float(alpha - base_alpha))
        out = {
            "theta_left": float(angle + alpha),
            "theta_right": float(angle - alpha),
            "obs_position": np.array([dx, dy], dtype=float),
            "half_angle": float(alpha),
            "base_half_angle": float(base_alpha),
            "civo_delta": float(civo_delta),
            "full_blockage": bool(alpha >= math.pi - 1e-9),
            "dist": float(dist),
            "R_sum": float(r_sum),
        }
        out.update(extra)
        return out

    def _refresh_civo_cones(self) -> None:
        self.civo_cones.clear()
        if not self._civo_active() or not self.obstacle_estimates:
            return
        for obs_id, est in self.obstacle_estimates.items():
            vo = self._compute_civo_single(int(obs_id), est)
            if vo is not None:
                self.civo_cones[int(obs_id)] = vo

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)

    def _angle_in_interval(self, angle: float, theta_right: float, theta_left: float) -> bool:
        angle = self._wrap_pi(angle)
        right = self._wrap_pi(theta_right)
        left = self._wrap_pi(theta_left)
        if right <= left:
            return bool(right <= angle <= left)
        return bool(angle >= right or angle <= left)

    def _velocity_inside_civo_for_est(self, u_own: float, v_own: float, est: Dict, vo: dict) -> bool:
        if not est:
            return False
        v_rel = np.array(
            [float(u_own) - float(est.get("vx", 0.0)), float(v_own) - float(est.get("vy", 0.0))],
            dtype=float,
        )
        if float(np.linalg.norm(v_rel)) < 1e-9:
            return False
        if bool(vo.get("full_blockage", False)):
            return bool(np.dot(vo["obs_position"], v_rel) > 0.0)
        angle = math.atan2(float(v_rel[1]), float(v_rel[0]))
        in_angle = self._angle_in_interval(angle, float(vo["theta_right"]), float(vo["theta_left"]))
        is_closing = bool(np.dot(vo["obs_position"], v_rel) > 0.0)
        return bool(in_angle and is_closing)

    def _velocity_inside_civo(self, u_own: float, v_own: float, obs_id: int, vo: dict) -> bool:
        est = self.obstacle_estimates.get(obs_id)
        return self._velocity_inside_civo_for_est(u_own, v_own, est, vo) if est else False

    def _civo_clearance_for_est(self, est: Dict, vo: dict) -> float:
        dist = float(vo.get("dist", math.hypot(float(est.get("dx", 0.0)), float(est.get("dy", 0.0)))))
        r_sum = float(vo.get("R_sum", self.R_usv + float(est.get("radius", self.dyn_radius))))
        d_min = float(vo.get("civo_d_min", dist))
        return float(d_min - r_sum)

    def _civo_shield_cone_is_gated(self, obs_id: int) -> bool:
        est = self.obstacle_estimates.get(obs_id)
        if not est:
            return False
        dist = math.hypot(float(est.get("dx", 0.0)), float(est.get("dy", 0.0)))
        tcpa = self._civo_tcpa_for_velocity(float(self.ship_state[3]), float(self.ship_state[4]), obs_id)
        in_time_window = math.isinf(tcpa) or 0.0 < tcpa <= self.civo_shield_tcpa_horizon
        return bool(dist <= self.civo_shield_distance and in_time_window)

    def _civo_shield_obstacle_ids(self) -> tuple[int, ...]:
        obs_ids = tuple(sorted(int(obs_id) for obs_id in self.obstacle_estimates))
        use_gate = (
            self.civo_shield_method == "corecbf"
            if self.civo_shield_gate_mode == "native"
            else self.civo_shield_gate_mode == "distance_tcpa"
        )
        if not use_gate:
            return obs_ids
        return tuple(
            obs_id
            for obs_id in obs_ids
            if self._civo_shield_cone_is_gated(obs_id)
        )

    def _civo_tcpa_for_velocity(self, u_own: float, v_own: float, obs_id: int) -> float:
        est = self.obstacle_estimates.get(obs_id)
        if not est:
            return float("inf")
        dx = float(est.get("dx", 0.0))
        dy = float(est.get("dy", 0.0))
        vx_rel = float(est.get("vx", 0.0)) - float(u_own)
        vy_rel = float(est.get("vy", 0.0)) - float(v_own)
        v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel
        if v_rel_sq < 1e-9:
            return float("inf")
        return float(-(dx * vx_rel + dy * vy_rel) / v_rel_sq)

    def _count_velocity_in_civo(self, u_own: float, v_own: float) -> tuple[int, dict[int, bool]]:
        inside_by_id: dict[int, bool] = {}
        n_inside = 0
        for obs_id, vo in self.civo_cones.items():
            inside = self._velocity_inside_civo(u_own, v_own, obs_id, vo)
            inside_by_id[int(obs_id)] = bool(inside)
            if inside:
                n_inside += 1
        return n_inside, inside_by_id

    def _collect_civo_metrics(self, u_own: float, v_own: float) -> dict[str, float | int | bool]:
        deltas = [float(vo.get("civo_delta", 0.0)) for vo in self.civo_cones.values()]
        clearances = [
            self._civo_clearance_for_est(self.obstacle_estimates[obs_id], vo)
            for obs_id, vo in self.civo_cones.items()
            if obs_id in self.obstacle_estimates
        ]
        n_inside, _ = self._count_velocity_in_civo(u_own, v_own)
        return {
            "civo_enabled": bool(self.civo_enabled),
            "civo_shield_enabled": bool(self.civo_shield_enabled),
            "civo_confidence": float(self.civo_confidence),
            "civo_cone_count": int(len(self.civo_cones)),
            "civo_active_count": int(sum(1 for d in deltas if d > 1e-8)),
            "civo_delta_max": float(np.max(deltas)) if deltas else 0.0,
            "civo_delta_sum": float(np.sum(deltas)) if deltas else 0.0,
            "civo_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
            "civo_clearance_min": float(np.min(clearances)) if clearances else float("nan"),
            "civo_inside_count": int(n_inside),
        }

    def _action_to_prop_speeds(self, action: np.ndarray) -> tuple[float, float]:
        raw_surge = float(np.clip(action[0], -1.0, 1.0))
        raw_yaw = float(np.clip(action[1], -1.0, 1.0))
        n_max = float(self.usv_params["n_max"])
        n_min = float(self.usv_params["n_min"])
        n_common = (raw_surge * n_max if raw_surge >= 0 else raw_surge * abs(n_min))
        n_common *= 0.70
        n_diff_max = min(n_max, abs(n_min))
        n_diff = raw_yaw * n_diff_max
        n_port = np.clip(n_common + n_diff, n_min, n_max)
        n_stbd = np.clip(n_common - n_diff, n_min, n_max)
        return float(n_port), float(n_stbd)

    def _prop_speeds_to_action(self, n_port: float, n_stbd: float) -> np.ndarray:
        n_max = float(self.usv_params["n_max"])
        n_min = float(self.usv_params["n_min"])
        n_common = 0.5 * (float(n_port) + float(n_stbd))
        n_diff = 0.5 * (float(n_port) - float(n_stbd))
        surge_den = 0.70 * (n_max if n_common >= 0.0 else abs(n_min))
        raw_surge = n_common / max(surge_den, 1e-9)
        raw_yaw = n_diff / max(min(n_max, abs(n_min)), 1e-9)
        return np.array([np.clip(raw_surge, -1.0, 1.0), np.clip(raw_yaw, -1.0, 1.0)], dtype=np.float32)

    def _prop_speed_for_thrust(self, thrust: float) -> float:
        thrust = float(thrust)
        if thrust >= 0.0:
            return math.sqrt(thrust / max(float(self.usv_params["k_pos"]), 1e-12))
        return -math.sqrt((-thrust) / max(float(self.usv_params["k_neg"]), 1e-12))

    def _thrust_min(self) -> float:
        return float(prop_thrust(float(self.usv_params["n_min"]), float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])))

    def _thrust_max(self) -> float:
        return float(prop_thrust(float(self.usv_params["n_max"]), float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])))

    def _action_to_thrusts(self, action: np.ndarray) -> np.ndarray:
        n_port, n_stbd = self._action_to_prop_speeds(action)
        return np.array(
            [
                prop_thrust(n_port, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])),
                prop_thrust(n_stbd, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"])),
            ],
            dtype=float,
        )

    def _action_to_control(self, action: np.ndarray) -> np.ndarray:
        thrusts = self._action_to_thrusts(action)
        y_p = float(self.usv_params["y_pontoon"])
        return np.array([thrusts[0] + thrusts[1], y_p * (thrusts[0] - thrusts[1])], dtype=float)

    def _control_to_thrusts(self, control: np.ndarray) -> np.ndarray:
        tau_u, tau_r = np.asarray(control, dtype=float).reshape(-1)[:2]
        y_p = max(abs(float(self.usv_params["y_pontoon"])), 1e-9)
        return np.array(
            [
                0.5 * (float(tau_u) + float(tau_r) / y_p),
                0.5 * (float(tau_u) - float(tau_r) / y_p),
            ],
            dtype=float,
        )

    def _thrusts_to_prop_speeds(self, thrusts: np.ndarray) -> tuple[float, float]:
        n_min = float(self.usv_params["n_min"])
        n_max = float(self.usv_params["n_max"])
        n_port = np.clip(self._prop_speed_for_thrust(float(thrusts[0])), n_min, n_max)
        n_stbd = np.clip(self._prop_speed_for_thrust(float(thrusts[1])), n_min, n_max)
        return float(n_port), float(n_stbd)

    def _apply_tau_u_rate_limit(
        self,
        n_port: float,
        n_stbd: float,
    ) -> tuple[float, float, float, float, bool]:
        k_pos = float(self.usv_params["k_pos"])
        k_neg = float(self.usv_params["k_neg"])
        thrust_port = float(prop_thrust(n_port, k_pos, k_neg))
        thrust_stbd = float(prop_thrust(n_stbd, k_pos, k_neg))
        y_p = max(abs(float(self.usv_params["y_pontoon"])), 1e-9)
        tau_u_ref = thrust_port + thrust_stbd
        tau_r_ref = y_p * (thrust_port - thrust_stbd)
        if self.actuator_tau_u_dot_max is None:
            return float(n_port), float(n_stbd), tau_u_ref, tau_r_ref, False

        max_change = self.actuator_tau_u_dot_max * float(self.dt)
        tau_u_min = max(2.0 * self._thrust_min(), self._last_tau_u_cmd - max_change)
        tau_u_max = min(2.0 * self._thrust_max(), self._last_tau_u_cmd + max_change)
        tau_u = float(np.clip(tau_u_ref, tau_u_min, tau_u_max))
        tau_r_max = y_p * max(
            0.0,
            min(tau_u - 2.0 * self._thrust_min(), 2.0 * self._thrust_max() - tau_u),
        )
        tau_r = float(np.clip(tau_r_ref, -tau_r_max, tau_r_max))
        limited = not (
            math.isclose(tau_u, tau_u_ref, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(tau_r, tau_r_ref, rel_tol=0.0, abs_tol=1e-9)
        )
        applied = self._control_to_thrusts(np.array([tau_u, tau_r], dtype=float))
        limited_port, limited_stbd = self._thrusts_to_prop_speeds(applied)
        return limited_port, limited_stbd, tau_u_ref, tau_r_ref, limited

    def _propeller_slew_command(
        self,
        requested_port: float,
        requested_stbd: float,
        previous_port: float,
        previous_stbd: float,
    ) -> tuple[float, float]:
        if self.actuator_n_dot_max is None or self.actuator_n_dot_max <= 0.0:
            return float(requested_port), float(requested_stbd)

        max_delta = float(self.actuator_n_dot_max) * float(self.dt)
        n_port_cmd = float(previous_port + np.clip(float(requested_port) - previous_port, -max_delta, max_delta))
        n_stbd_cmd = float(previous_stbd + np.clip(float(requested_stbd) - previous_stbd, -max_delta, max_delta))
        return n_port_cmd, n_stbd_cmd

    def _apply_propeller_slew(self, n_port: float, n_stbd: float) -> tuple[float, float]:
        n_port_cmd, n_stbd_cmd = self._propeller_slew_command(
            n_port,
            n_stbd,
            self._last_n_port_cmd,
            self._last_n_stbd_cmd,
        )
        self._last_n_port_cmd = n_port_cmd
        self._last_n_stbd_cmd = n_stbd_cmd
        return n_port_cmd, n_stbd_cmd

    def _predict_next_augmented_state_for_prop_speeds(
        self,
        n_port: float,
        n_stbd: float,
    ) -> tuple[np.ndarray, float, float, float]:
        n_port, n_stbd = self._propeller_slew_command(
            n_port,
            n_stbd,
            self._last_n_port_cmd,
            self._last_n_stbd_cmd,
        )
        predicted = update_usv_full_model(
            self.ship_state.copy(),
            n_port,
            n_stbd,
            self.dt,
            self.usv_params,
        )
        predicted[2] = normalize_angle_0_to_2pi(predicted[2])
        tau_u = float(
            prop_thrust(n_port, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"]))
            + prop_thrust(n_stbd, float(self.usv_params["k_pos"]), float(self.usv_params["k_neg"]))
        )
        return predicted, n_port, n_stbd, tau_u

    def _predict_next_augmented_state_for_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        return self._predict_next_augmented_state_for_prop_speeds(*self._action_to_prop_speeds(action))

    def _predict_next_state_for_thrusts(self, thrusts: np.ndarray) -> np.ndarray:
        return self._predict_states_for_thrusts(thrusts, steps=1)[0]

    def _predict_states_for_thrusts(
        self, thrusts: np.ndarray, *, steps: int
    ) -> np.ndarray:
        if steps not in (1, 2):
            raise ValueError("thrust rollout supports one or two steps")
        n_port, n_stbd = self._thrusts_to_prop_speeds(thrusts)
        state = self.ship_state.copy()
        states = []
        for _ in range(steps):
            state = update_usv_full_model(
                state, n_port, n_stbd, self.dt, self.usv_params
            )
            state[2] = normalize_angle_0_to_2pi(state[2])
            states.append(state.copy())
        return np.asarray(states)

    def _predict_next_state_for_action(self, action: np.ndarray) -> np.ndarray:
        return self._predict_next_augmented_state_for_action(action)[0]

    def _position_in_free_space(self, x: float, y: float) -> bool:
        if not in_bounds(float(x), float(y), self.W, self.H):
            return False
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < self.W and 0 <= yi < self.H and self.grid_map[yi, xi] == 1:
            return False
        return True

    def _target_world_prediction(self, est: Dict) -> tuple[np.ndarray, np.ndarray]:
        psi = float(self.ship_state[2])
        c, s = math.cos(psi), math.sin(psi)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        rel = np.array([float(est.get("dx", 0.0)), float(est.get("dy", 0.0))], dtype=float)
        vel = np.array([float(est.get("vx", 0.0)), float(est.get("vy", 0.0))], dtype=float)
        pos_w = np.asarray(self.ship_state[:2], dtype=float) + rot @ rel
        vel_w = rot @ vel
        return pos_w, pos_w + vel_w * float(self.dt)

    def _predict_estimate_for_state(self, next_state: np.ndarray, est: Dict) -> Dict:
        pos_now, pos_next = self._target_world_prediction(est)
        dt = max(float(self.dt), 1e-9)
        vel_w = (pos_next - pos_now) / dt
        psi = float(next_state[2])
        c, s = math.cos(-psi), math.sin(-psi)
        rel_w = pos_next - np.asarray(next_state[:2], dtype=float)
        next_est = dict(est)
        next_est["dx"] = float(c * rel_w[0] - s * rel_w[1])
        next_est["dy"] = float(s * rel_w[0] + c * rel_w[1])
        next_est["vx"] = float(c * vel_w[0] - s * vel_w[1])
        next_est["vy"] = float(s * vel_w[0] + c * vel_w[1])
        return next_est

    def _civo_corecbf_terms(self, obs_id: int) -> dict:
        est = self.obstacle_estimates.get(obs_id)
        if not est:
            raise RuntimeError(f"missing obstacle estimate {obs_id}")
        target_position, target_position_next = self._target_world_prediction(est)
        target_velocity = (target_position_next - target_position) / float(self.dt)
        kwargs = dict(
            ship_state=self.ship_state,
            target_position=target_position,
            target_velocity=target_velocity,
            safety_distance=self.corecbf_safety_distance,
            surge_accel=self.corecbf_surge_accel,
            turn_accel=self.corecbf_turn_accel,
            turn_direction=self.corecbf_turn_direction,
            usv_params=self.usv_params,
        )
        if self.corecbf_variant == "deterministic":
            return corecbf_terms(**kwargs)

        p_credible = self._credible_covariance_for_obstacle(obs_id)
        try:
            geometry = credible_geometry_scale(
                relative_position=target_position - np.asarray(self.ship_state[:2], dtype=float),
                position_covariance=p_credible[:2, :2],
                safety_distance=self.corecbf_safety_distance,
                confidence_scale=self._civo_sigma_scale,
            )
        except CredibleGeometryDomainError as exc:
            terms = corecbf_terms(**kwargs)
            terms.update(
                {
                    "credible_geometry_fallback": True,
                    "credible_geometry_fallback_reason": str(exc),
                }
            )
            return terms
        terms = credibility_aware_corecbf_terms(
            **kwargs,
            geometry_scale=geometry["lambda"],
        )
        terms.update(
            {
                "credible_alpha": geometry["alpha"],
                "credible_d_min": geometry["d_min"],
                "credible_e_parallel": geometry["e_parallel"],
                "credible_e_perp": geometry["e_perp"],
                "credible_geometry_fallback": False,
                "credible_geometry_fallback_reason": "",
            }
        )
        return terms

    def _corecbf_reference_control(self, action: np.ndarray) -> np.ndarray:
        nominal_action = np.clip(np.asarray(action, dtype=float).reshape(-1)[:2], -1.0, 1.0)
        nominal_control = self._action_to_control(nominal_action)
        if self.corecbf_variant == "credibility_colregs":
            self._update_colregs_signal()
            nominal_control = colregs_reference_control(
                nominal_control=nominal_control,
                aggregate_duty=self._last_colregs_aggregate_duty,
                turn_accel=self.corecbf_turn_accel,
                yaw_control_gain=1.0 / float(self.usv_params["m33"]),
                reference_scale=self.corecbf_colregs_reference_scale,
            )
        self._last_colregs_reference_control = np.asarray(nominal_control, dtype=float).copy()
        return self._last_colregs_reference_control.copy()

    def _credible_colregs_action(self, policy_action: np.ndarray) -> np.ndarray:
        reference_control = self._corecbf_reference_control(policy_action)
        thrusts = np.clip(
            self._control_to_thrusts(reference_control),
            self._thrust_min(),
            self._thrust_max(),
        )
        n_port, n_stbd = self._thrusts_to_prop_speeds(thrusts)
        self._last_civo_prop_override = (n_port, n_stbd)
        return self._prop_speeds_to_action(n_port, n_stbd)

    def _unconstrained_reference_action(
        self,
        policy_action: np.ndarray,
        policy_props: np.ndarray,
    ) -> np.ndarray:
        if self.corecbf_variant != "credibility_colregs":
            return policy_action
        candidate = self._credible_colregs_action(policy_action)
        applied_props = np.asarray(self._last_civo_prop_override, dtype=float)
        physical_delta = float(np.linalg.norm(applied_props - policy_props))
        self._last_civo_shield_changed = bool(physical_delta > 1e-8)
        self._last_civo_shield_delta = (
            float(np.linalg.norm(candidate - policy_action))
            if self._last_civo_shield_changed
            else 0.0
        )
        return candidate.astype(np.float32)

    def _solve_civo_corecbf_qp(self, action: np.ndarray, obs_ids) -> dict:
        obs_ids = tuple(sorted(int(obs_id) for obs_id in obs_ids))
        if not obs_ids:
            raise ValueError("CoReCBF QP requires at least one obstacle")
        nominal_control = self._corecbf_reference_control(action)
        terms = [self._civo_corecbf_terms(obs_id) for obs_id in obs_ids]
        fallbacks = [
            (obs_id, str(item.get("credible_geometry_fallback_reason", "")))
            for obs_id, item in zip(obs_ids, terms)
            if bool(item.get("credible_geometry_fallback", False))
        ]
        solved = solve_corecbf_qp(
            nominal_control=nominal_control,
            h=np.array([item["H"] for item in terms]),
            lf_h=np.array([item["LfH"] for item in terms]),
            lg_h=np.vstack([item["LgH"] for item in terms]),
            cbf_gain=self.corecbf_gain,
            y_p=abs(float(self.usv_params["y_pontoon"])),
            thrust_lower=self._thrust_min(),
            thrust_upper=self._thrust_max(),
            weights=np.array([self.corecbf_tau_u_weight, self.corecbf_tau_r_weight]),
            slack_weight=self.corecbf_slack_weight,
            eps_abs=self.corecbf_osqp_eps_abs,
            eps_rel=self.corecbf_osqp_eps_rel,
            max_iter=self.corecbf_osqp_max_iter,
            polishing=self.corecbf_osqp_polishing,
            separable_slack=self.corecbf_separable_slack_enabled,
        )
        self._last_credible_geometry_fallback_ids = tuple(item[0] for item in fallbacks)
        self._last_credible_geometry_fallback_reasons = tuple(item[1] for item in fallbacks)
        if fallbacks:
            self._credible_geometry_fallback_steps += 1
            self._credible_geometry_fallback_obstacle_steps += len(fallbacks)
        h_values = np.array([item["H"] for item in terms])
        worst_index = int(np.argmin(h_values))
        self._last_civo_constraint_count = len(obs_ids)
        self._last_civo_obstacle_id = obs_ids[worst_index]
        self._last_civo_h = float(h_values[worst_index])
        self._last_civo_residual = float(np.min(solved["residuals"]))
        self._last_civo_slack = float(solved["slack"])
        self._last_civo_qp_status = str(solved["status"])
        self._last_civo_qp_solver = str(solved["solver"])
        self._last_civo_qp_iter = int(solved["iterations"])
        return solved

    def _solve_civo_relaxed_vo_cbf_qp(
        self, action: np.ndarray
    ) -> dict:
        nominal_thrusts = self._action_to_thrusts(action)
        thrust_lower = self._thrust_min()
        thrust_upper = self._thrust_max()
        _, control_to_thrust, braking_accel, _ = otter_acceleration_maps(
            m11=self.usv_params["m11"],
            m33=self.usv_params["m33"],
            y_p=abs(float(self.usv_params["y_pontoon"])),
            thrust_lower=thrust_lower,
            thrust_upper=thrust_upper,
        )

        vo_items: list[tuple[int, dict]] = []
        hard_items: list[tuple[int, dict]] = []
        domain_exit_ids: list[int] = []
        for obs_id in self._civo_shield_obstacle_ids():
            est = self.obstacle_estimates[obs_id]
            target_position, target_position_next = (
                self._target_world_prediction(est)
            )
            target_velocity = (
                target_position_next - target_position
            ) / max(float(self.dt), 1e-9)
            common = dict(
                ship_state=self.ship_state,
                target_position=target_position,
                target_velocity=target_velocity,
                safety_distance=self.corecbf_safety_distance,
                usv_params=self.usv_params,
            )
            hard = hard_collision_cbf_terms(
                **common,
                braking_accel=braking_accel,
            )
            try:
                vo = relaxed_vo_cbf_terms(**common)
            except RelaxedVOCBFDomainError:
                vo = None
                domain_exit_ids.append(int(obs_id))
            if vo is not None:
                vo_items.append((int(obs_id), vo))
            if hard is not None:
                hard_items.append((int(obs_id), hard))

        def values(
            items: list[tuple[int, dict]], key: str
        ) -> np.ndarray:
            return np.array([item[key] for _, item in items], dtype=float)

        def rows(items: list[tuple[int, dict]]) -> np.ndarray:
            return (
                np.vstack([item["LgH_thrust"] for _, item in items])
                if items
                else np.empty((0, 2), dtype=float)
            )

        self._last_civo_mechanism.update(
            {
                "vo_cbf_active_vo_count": len(vo_items),
                "vo_cbf_active_hard_count": len(hard_items),
                "vo_cbf_domain_exit_count": len(domain_exit_ids),
                "vo_cbf_domain_exit_ids": ";".join(
                    str(obs_id) for obs_id in domain_exit_ids
                ),
            }
        )
        self._vo_cbf_ep_domain_exit_count += len(domain_exit_ids)
        solved = solve_relaxed_vo_cbf_qp(
            nominal_thrusts=nominal_thrusts,
            control_to_thrust=control_to_thrust,
            vo_h=values(vo_items, "H"),
            vo_lf_h=values(vo_items, "LfH"),
            vo_lg_thrust=rows(vo_items),
            vo_ttc=values(vo_items, "ttc"),
            hard_h=values(hard_items, "H"),
            hard_lf_h=values(hard_items, "LfH"),
            hard_lg_thrust=rows(hard_items),
            alpha_vo=self.vo_cbf_alpha_vo,
            alpha_c=self.vo_cbf_alpha_c,
            k_u=self.vo_cbf_k_u,
            k_vo=self.vo_cbf_k_vo,
            thrust_lower=thrust_lower,
            thrust_upper=thrust_upper,
            eps_abs=self.corecbf_osqp_eps_abs,
            eps_rel=self.corecbf_osqp_eps_rel,
            max_iter=self.corecbf_osqp_max_iter,
            polishing=self.corecbf_osqp_polishing,
        )
        solved["nominal_thrusts"] = nominal_thrusts
        solved["vo_h"] = values(vo_items, "H")
        solved["hard_h"] = values(hard_items, "H")
        solved["vo_obstacle_ids"] = tuple(
            obs_id for obs_id, _ in vo_items
        )
        solved["hard_obstacle_ids"] = tuple(
            obs_id for obs_id, _ in hard_items
        )
        solved["vo_domain_exit_ids"] = tuple(domain_exit_ids)
        return solved

    def _cbf_vo_rows(
        self, action: np.ndarray
    ) -> dict:
        nominal = np.clip(
            self._action_to_thrusts(action),
            self._thrust_min(),
            self._thrust_max(),
        )
        thrust_lower, thrust_upper = self._thrust_min(), self._thrust_max()
        _, control_to_thrust, braking_accel, _ = otter_acceleration_maps(
            m11=self.usv_params["m11"],
            m33=self.usv_params["m33"],
            y_p=abs(float(self.usv_params["y_pontoon"])),
            thrust_lower=thrust_lower,
            thrust_upper=thrust_upper,
        )
        dt = float(self.dt)
        targets = {}
        current_domain_exit_ids = []
        for obs_id in self._civo_shield_obstacle_ids():
            position, position_next = self._target_world_prediction(
                self.obstacle_estimates[obs_id]
            )
            velocity = (position_next - position) / max(dt, 1e-9)
            common = dict(
                ship_state=self.ship_state,
                target_position=position,
                target_velocity=velocity,
                safety_distance=self.corecbf_safety_distance,
            )
            hard_h = hard_collision_barrier_value(
                **common, braking_accel=braking_accel
            )
            try:
                vo = relaxed_vo_cbf_terms(
                    **common, usv_params=self.usv_params
                )
            except RelaxedVOCBFDomainError:
                vo = None
                current_domain_exit_ids.append(int(obs_id))
            targets[int(obs_id)] = {
                "position": position,
                "velocity": velocity,
                "vo_h": None if vo is None else float(vo["H"]),
                "ttc": None if vo is None else float(vo["ttc"]),
                "hard_h": hard_h,
            }

        def values(thrusts: np.ndarray) -> tuple[dict, dict]:
            states = self._predict_states_for_thrusts(
                thrusts, steps=CBF_VO_PREDICTION_STEPS
            )
            vo_values, hard_values = {}, {}
            for obs_id, target in targets.items():
                for stage, state in enumerate(states, start=1):
                    target_position = (
                        target["position"] + stage * dt * target["velocity"]
                    )
                    common = dict(
                        ship_state=state,
                        target_position=target_position,
                        target_velocity=target["velocity"],
                        safety_distance=self.corecbf_safety_distance,
                    )
                    hard_values[(obs_id, stage)] = (
                        hard_collision_barrier_value(
                            **common, braking_accel=braking_accel
                        )
                        - math.exp(-self.vo_cbf_alpha_c * stage * dt)
                        * target["hard_h"]
                    )
                    if target["vo_h"] is None:
                        continue
                    try:
                        future_h = vo_barrier_value(**common)
                    except RelaxedVOCBFDomainError:
                        continue
                    vo_values[(obs_id, stage)] = (
                        future_h
                        - math.exp(-self.vo_cbf_alpha_vo * stage * dt)
                        * target["vo_h"]
                    )
            return vo_values, hard_values

        base_vo, base_hard = values(nominal)
        vo_keys, hard_keys = tuple(base_vo), tuple(base_hard)

        def finite_difference(fraction: float) -> tuple[np.ndarray, np.ndarray]:
            epsilon = fraction * (thrust_upper - thrust_lower)
            plus_values, minus_values = [], []
            plus_thrusts, minus_thrusts = [], []
            for axis in range(2):
                plus, minus = nominal.copy(), nominal.copy()
                plus[axis] = min(thrust_upper, plus[axis] + epsilon)
                minus[axis] = max(thrust_lower, minus[axis] - epsilon)
                plus_thrusts.append(plus)
                minus_thrusts.append(minus)
                plus_values.append(values(plus))
                minus_values.append(values(minus))

            def rows_for(
                base: dict, keys: tuple, value_index: int
            ) -> np.ndarray:
                rows = np.empty((len(keys), 2), dtype=float)
                for row_index, key in enumerate(keys):
                    for axis in range(2):
                        plus = plus_values[axis][value_index]
                        minus = minus_values[axis][value_index]
                        x_plus = plus_thrusts[axis][axis]
                        x_minus = minus_thrusts[axis][axis]
                        if key in plus and key in minus and x_plus > x_minus:
                            derivative = (plus[key] - minus[key]) / (
                                x_plus - x_minus
                            )
                        elif key in plus and x_plus > nominal[axis]:
                            derivative = (plus[key] - base[key]) / (
                                x_plus - nominal[axis]
                            )
                        elif key in minus and nominal[axis] > x_minus:
                            derivative = (base[key] - minus[key]) / (
                                nominal[axis] - x_minus
                            )
                        else:
                            raise RelaxedVOCBFNoVerifiedAction(
                                "predictive VO-CBF finite difference left "
                                "the barrier domain"
                            )
                        rows[row_index, axis] = derivative
                return rows

            return (
                rows_for(base_vo, vo_keys, 0),
                rows_for(base_hard, hard_keys, 1),
            )

        vo_rows, hard_rows = finite_difference(
            VO_CBF_FINITE_DIFFERENCE_FRACTION
        )
        half_vo_rows, half_hard_rows = finite_difference(
            0.5 * VO_CBF_FINITE_DIFFERENCE_FRACTION
        )
        input_rows = np.vstack((vo_rows, hard_rows))
        half_input_rows = np.vstack((half_vo_rows, half_hard_rows))
        control_rows = input_rows @ control_to_thrust
        active_vo_ids = tuple(dict.fromkeys(key[0] for key in vo_keys))
        slack_index = {obs_id: index for index, obs_id in enumerate(active_vo_ids)}
        rank = (
            int(np.linalg.matrix_rank(control_rows, tol=1e-10))
            if control_rows.size
            else 0
        )
        yaw_sensitivity = (
            float(np.max(np.abs(control_rows[:, 1])))
            if control_rows.size
            else 0.0
        )
        return {
            "nominal_thrusts": nominal,
            "control_to_thrust": control_to_thrust,
            "vo_offset": np.array(
                [base_vo[key] for key in vo_keys]
            )
            - vo_rows @ nominal,
            "vo_input_rows": vo_rows,
            "vo_slack_ids": np.array(
                [slack_index[key[0]] for key in vo_keys], dtype=int
            ),
            "vo_ttc": np.array(
                [targets[obs_id]["ttc"] for obs_id in active_vo_ids]
            ),
            "hard_offset": np.array(
                [base_hard[key] for key in hard_keys]
            )
            - hard_rows @ nominal,
            "hard_input_rows": hard_rows,
            "vo_row_keys": vo_keys,
            "hard_row_keys": hard_keys,
            "input_rows": input_rows,
            "half_epsilon_input_rows": half_input_rows,
            "safety_row_rank": rank,
            "yaw_sensitivity_max": yaw_sensitivity,
            "vo_domain_exit_keys": tuple(
                (obs_id, stage)
                for obs_id, target in targets.items()
                if target["vo_h"] is not None
                for stage in range(1, CBF_VO_PREDICTION_STEPS + 1)
                if (obs_id, stage) not in base_vo
            ),
            "vo_current_domain_exit_ids": tuple(current_domain_exit_ids),
            "targets": targets,
            "braking_accel": braking_accel,
        }

    def _solve_cbf_vo_qp(
        self, action: np.ndarray
    ) -> dict:
        rows = self._cbf_vo_rows(action)
        domain_exit_labels = [
            str(obs_id) for obs_id in rows["vo_current_domain_exit_ids"]
        ] + [
            f"{obs_id}:{stage}" for obs_id, stage in rows["vo_domain_exit_keys"]
        ]
        domain_exit_count = len(domain_exit_labels)
        active_vo_ids = tuple(
            dict.fromkeys(key[0] for key in rows["vo_row_keys"])
        )
        self._last_civo_mechanism.update(
            {
                "vo_cbf_active_vo_count": len(active_vo_ids),
                "vo_cbf_active_hard_count": len(rows["targets"]),
                "vo_cbf_active_vo_row_count": len(rows["vo_row_keys"]),
                "vo_cbf_active_hard_row_count": len(rows["hard_row_keys"]),
                "vo_cbf_domain_exit_count": domain_exit_count,
                "vo_cbf_domain_exit_ids": ";".join(domain_exit_labels),
                "vo_cbf_predictive_domain_exit_count": domain_exit_count,
                "cbf_vo_prediction_steps": CBF_VO_PREDICTION_STEPS,
            }
        )
        self._vo_cbf_ep_domain_exit_count += domain_exit_count
        if self.actuator_tau_u_dot_max is None:
            thrust_sum_lower = None
            thrust_sum_upper = None
        else:
            max_change = self.actuator_tau_u_dot_max * float(self.dt)
            thrust_sum_lower = max(
                2.0 * self._thrust_min(),
                self._last_tau_u_cmd - max_change,
            )
            thrust_sum_upper = min(
                2.0 * self._thrust_max(),
                self._last_tau_u_cmd + max_change,
            )
        solved = solve_relaxed_affine_qp(
            nominal_thrusts=rows["nominal_thrusts"],
            control_to_thrust=rows["control_to_thrust"],
            vo_offset=rows["vo_offset"],
            vo_input_rows=rows["vo_input_rows"],
            vo_slack_ids=rows["vo_slack_ids"],
            vo_ttc=rows["vo_ttc"],
            hard_offset=rows["hard_offset"],
            hard_input_rows=rows["hard_input_rows"],
            k_u=self.vo_cbf_k_u,
            k_vo=self.vo_cbf_k_vo,
            thrust_lower=self._thrust_min(),
            thrust_upper=self._thrust_max(),
            thrust_sum_lower=thrust_sum_lower,
            thrust_sum_upper=thrust_sum_upper,
            eps_abs=self.corecbf_osqp_eps_abs,
            eps_rel=self.corecbf_osqp_eps_rel,
            max_iter=self.corecbf_osqp_max_iter,
            polishing=self.corecbf_osqp_polishing,
        )
        thrusts = np.asarray(solved["thrusts"], dtype=float)
        states = self._predict_states_for_thrusts(
            thrusts, steps=CBF_VO_PREDICTION_STEPS
        )

        def exact_value(key: tuple[int, int], *, hard: bool) -> float:
            obs_id, stage = key
            target = rows["targets"][obs_id]
            common = dict(
                ship_state=states[stage - 1],
                target_position=(
                    target["position"]
                    + stage * float(self.dt) * target["velocity"]
                ),
                target_velocity=target["velocity"],
                safety_distance=self.corecbf_safety_distance,
            )
            if hard:
                future_h = hard_collision_barrier_value(
                    **common, braking_accel=rows["braking_accel"]
                )
                return float(
                    future_h
                    - math.exp(
                        -self.vo_cbf_alpha_c * stage * float(self.dt)
                    )
                    * target["hard_h"]
                )
            future_h = vo_barrier_value(**common)
            return float(
                future_h
                - math.exp(
                    -self.vo_cbf_alpha_vo * stage * float(self.dt)
                )
                * target["vo_h"]
            )

        try:
            exact_vo = np.array(
                [exact_value(key, hard=False) for key in rows["vo_row_keys"]]
            )
            exact_hard = np.array(
                [exact_value(key, hard=True) for key in rows["hard_row_keys"]]
            )
        except (RelaxedVOCBFDomainError, ValueError) as exc:
            raise RelaxedVOCBFNoVerifiedAction(
                f"nonlinear predictive barrier evaluation failed: {exc}"
            ) from exc

        slacks = np.asarray(solved["slacks"], dtype=float)
        exact_relaxed_vo = exact_vo + slacks[rows["vo_slack_ids"]]
        tolerance = max(1e-9, 10.0 * self.corecbf_osqp_eps_abs)
        if np.any(exact_relaxed_vo < -tolerance) or np.any(
            exact_hard < -tolerance
        ):
            raise RelaxedVOCBFNoVerifiedAction(
                "nonlinear predictive VO-CBF residual verification failed"
            )

        affine_vo = rows["vo_offset"] + rows["vo_input_rows"] @ thrusts
        affine_hard = (
            rows["hard_offset"] + rows["hard_input_rows"] @ thrusts
        )
        errors = np.concatenate(
            (np.abs(exact_vo - affine_vo), np.abs(exact_hard - affine_hard))
        )
        solved.update(
            {
                "nominal_thrusts": rows["nominal_thrusts"],
                "vo_residuals": exact_vo,
                "relaxed_vo_residuals": exact_relaxed_vo,
                "hard_residuals": exact_hard,
                "affine_vo_residuals": affine_vo,
                "affine_hard_residuals": affine_hard,
                "predictive_residual_error_max": (
                    float(np.max(errors)) if errors.size else 0.0
                ),
                "vo_obstacle_ids": active_vo_ids,
                "hard_obstacle_ids": tuple(rows["targets"]),
                "vo_row_keys": rows["vo_row_keys"],
                "hard_row_keys": rows["hard_row_keys"],
                "vo_domain_exit_keys": rows["vo_domain_exit_keys"],
                "vo_h": np.array(
                    [rows["targets"][obs_id]["vo_h"] for obs_id in active_vo_ids]
                ),
                "hard_h": np.array(
                    [item["hard_h"] for item in rows["targets"].values()]
                ),
            }
        )
        return solved

    def _shield_action_relaxed_vo_cbf(
        self,
        policy_action: np.ndarray,
        policy_props: np.ndarray,
    ) -> np.ndarray:
        try:
            if self.civo_shield_method == "cbf_vo":
                self._last_civo_mechanism[
                    "cbf_vo_prediction_steps"
                ] = CBF_VO_PREDICTION_STEPS
                solved = self._solve_cbf_vo_qp(
                    policy_action
                )
            else:
                solved = self._solve_civo_relaxed_vo_cbf_qp(policy_action)
        except RelaxedVOCBFNoVerifiedAction as exc:
            reason = str(exc)
            self._last_civo_qp_success = False
            self._last_civo_qp_status = f"nominal fallback: {reason}"
            self._last_civo_qp_solver = "osqp"
            self._last_civo_prop_override = None
            self._last_civo_mechanism.update(
                {
                    "vo_cbf_fallback": 1,
                    "vo_cbf_fallback_reason": reason,
                }
            )
            if (
                self.civo_shield_method == "cbf_vo"
                and "nonlinear predictive" in reason
            ):
                self._last_civo_mechanism[
                    "vo_cbf_predictive_rejection_count"
                ] = 1
            self._vo_cbf_ep_fallback_count += 1
            return policy_action.copy()
        thrusts = np.asarray(solved["thrusts"], dtype=float)
        nominal = np.asarray(solved["nominal_thrusts"], dtype=float)
        n_port, n_stbd = self._thrusts_to_prop_speeds(thrusts)
        candidate = self._prop_speeds_to_action(n_port, n_stbd)
        self._last_civo_prop_override = (n_port, n_stbd)
        self._last_civo_qp_success = True
        self._last_civo_qp_status = str(solved["status"])
        self._last_civo_qp_solver = str(solved["solver"])
        self._last_civo_qp_iter = int(solved["iterations"])
        self._last_civo_constraint_count = int(
            len(solved["vo_obstacle_ids"])
            + len(solved["hard_obstacle_ids"])
        )
        all_h = np.concatenate((solved["vo_h"], solved["hard_h"]))
        self._last_civo_h = (
            float(np.min(all_h)) if all_h.size else float("nan")
        )
        accepted = np.concatenate(
            (solved["relaxed_vo_residuals"], solved["hard_residuals"])
        )
        self._last_civo_residual = (
            float(np.min(accepted)) if accepted.size else float("nan")
        )
        slacks = np.asarray(solved["slacks"], dtype=float)
        self._last_civo_slack = (
            float(np.max(slacks)) if slacks.size else 0.0
        )
        physical_delta = float(
            np.linalg.norm(np.array([n_port, n_stbd]) - policy_props)
        )
        self._last_civo_shield_changed = bool(physical_delta > 1e-8)
        self._last_civo_shield_delta = (
            float(np.linalg.norm(candidate - policy_action))
            if self._last_civo_shield_changed
            else 0.0
        )

        thrust_delta = thrusts - nominal
        y_p = float(self.usv_params["y_pontoon"])
        self._last_civo_mechanism.update(
            {
                "vo_cbf_active_vo_count": int(
                    len(solved["vo_obstacle_ids"])
                ),
                "vo_cbf_active_hard_count": int(
                    len(solved["hard_obstacle_ids"])
                ),
                "vo_cbf_slack_sum": float(np.sum(slacks)),
                "vo_cbf_slack_max": (
                    float(np.max(slacks)) if slacks.size else 0.0
                ),
                "vo_cbf_vo_residual_min": (
                    float(np.min(solved["relaxed_vo_residuals"]))
                    if solved["relaxed_vo_residuals"].size
                    else float("nan")
                ),
                "vo_cbf_hard_residual_min": (
                    float(np.min(solved["hard_residuals"]))
                    if solved["hard_residuals"].size
                    else float("nan")
                ),
                "vo_cbf_delta_tau_u": float(
                    thrust_delta[0] + thrust_delta[1]
                ),
                "vo_cbf_delta_tau_r": float(
                    y_p * (thrust_delta[0] - thrust_delta[1])
                ),
                "vo_cbf_delta_thrust_common": float(
                    0.5 * (thrust_delta[0] + thrust_delta[1])
                ),
                "vo_cbf_delta_thrust_differential": float(
                    0.5 * (thrust_delta[0] - thrust_delta[1])
                ),
                "vo_cbf_safety_row_rank": int(
                    solved["safety_row_rank"]
                ),
                "vo_cbf_yaw_sensitivity_max": float(
                    solved["yaw_sensitivity_max"]
                ),
            }
        )
        if self.civo_shield_method == "cbf_vo":
            self._last_civo_mechanism.update(
                {
                    "vo_cbf_predictive_residual_error_max": float(
                        solved["predictive_residual_error_max"]
                    ),
                    "cbf_vo_prediction_steps": CBF_VO_PREDICTION_STEPS,
                }
            )
        return candidate.astype(np.float32)

    def _shield_action_civo(self, action: np.ndarray) -> np.ndarray:
        started = time.perf_counter_ns()
        policy_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0
        )
        policy_props = np.asarray(self._action_to_prop_speeds(policy_action), dtype=float)
        self._last_civo_shield_changed = False
        self._last_civo_shield_delta = 0.0
        self._last_civo_qp_success = False
        self._last_civo_qp_solver = "none"
        self._last_civo_qp_status = ""
        self._last_civo_qp_iter = 0
        self._last_civo_constraint_count = 0
        self._last_civo_obstacle_id = -1
        self._last_civo_h = float("nan")
        self._last_civo_residual = float("nan")
        self._last_civo_slack = 0.0
        self._last_civo_prop_override = None
        self._last_civo_mechanism = _empty_vo_cbf_diagnostics()
        self._last_credible_geometry_fallback_ids = ()
        self._last_credible_geometry_fallback_reasons = ()
        try:
            if not self.civo_shield_enabled:
                self._last_civo_qp_status = "shield disabled"
                return policy_action
            if self.civo_shield_method in {
                "relaxed_vo_cbf",
                "cbf_vo",
            }:
                return self._shield_action_relaxed_vo_cbf(
                    policy_action,
                    policy_props,
                )
            if not self.civo_cones:
                self._last_civo_qp_status = "no active cones"
                return self._unconstrained_reference_action(policy_action, policy_props)
            selected_ids = set(self._civo_shield_obstacle_ids())
            gated_cones = {
                obs_id: vo
                for obs_id, vo in self.civo_cones.items()
                if obs_id in selected_ids
            }
            if not gated_cones:
                self._last_civo_qp_status = "no gated cones"
                return self._unconstrained_reference_action(policy_action, policy_props)
            solved = self._solve_civo_corecbf_qp(
                policy_action, gated_cones
            )
            thrusts = np.asarray(solved["thrusts"], dtype=float)
            n_port, n_stbd = self._thrusts_to_prop_speeds(thrusts)
            candidate = self._prop_speeds_to_action(n_port, n_stbd)
            self._last_civo_prop_override = (n_port, n_stbd)
            self._last_civo_qp_success = True
            physical_delta = float(
                np.linalg.norm(np.array([n_port, n_stbd]) - policy_props)
            )
            self._last_civo_shield_changed = bool(physical_delta > 1e-8)
            self._last_civo_shield_delta = (
                float(np.linalg.norm(candidate - policy_action))
                if self._last_civo_shield_changed
                else 0.0
            )
            return candidate.astype(np.float32)
        except SafetyFilterRuntimeError:
            raise
        except Exception as exc:
            label = (
                "CoReCBF"
                if self.civo_shield_method == "corecbf"
                else (
                    "CBF-VO"
                    if self.civo_shield_method == "cbf_vo"
                    else "Direct Relaxed VO-CBF"
                )
            )
            reason = f"{type(exc).__name__}: {exc}"
            self._last_civo_qp_status = f"filter exception: {reason}"
            self._last_civo_mechanism["vo_cbf_failure_reason"] = reason
            self._last_civo_prop_override = None
            raise SafetyFilterRuntimeError(
                f"{label} failed: {reason}"
            ) from exc
        finally:
            elapsed_ns = time.perf_counter_ns() - started
            self._last_civo_shield_latency_ns = (
                int(elapsed_ns) if self.civo_shield_enabled else 0
            )
            self._last_civo_shield_latency_ms = (
                self._last_civo_shield_latency_ns / 1e6
            )
            self._civo_ep_latency_ms.append(
                self._last_civo_shield_latency_ms
            )

    # ==================================================================
    #   Helper methods
    # ==================================================================

    def seed(self, seed: Optional[int] = None):
        self.np_random, seeded = gymnasium.utils.seeding.np_random(seed)
        self._last_seed = int(seeded)
        self.rng = np.random.default_rng(self._last_seed)
        return [self._last_seed]

    def set_grid(self, grid: np.ndarray):
        self.grid_map = grid.copy()
        self.H, self.W = grid.shape
        self.obstacle_coords = np.argwhere(self.grid_map == 1)

    def set_dyn_traj(self, dyn_traj: np.ndarray):
        if not self.dynamic_obstacles:
            self.use_dyn_replay = False
            self.dyn_traj = None
            self.dyn_obs_num = 0
            self.dyn_pos = np.zeros((0, 4), dtype=np.float32)
            self.dyn_true_traj, self.dyn_est_traj = [], []
            return
        dyn_traj = self._normalize_dyn_traj(dyn_traj)
        n_obstacles = int(dyn_traj.shape[1])
        if n_obstacles > self.DYN_MAX:
            raise DynamicObstacleCapacityError(
                f"dyn_traj obstacle count {n_obstacles} exceeds environment capacity {self.DYN_MAX}; "
                "construct env with scenario directory to size capacity from the dataset"
            )
        self.dyn_traj = dyn_traj
        self.scenario_T = int(dyn_traj.shape[0])
        self.use_dyn_replay = True
        self.dyn_step = 0
        self.dyn_obs_num = n_obstacles
        self.dyn_pos = np.zeros((self.dyn_obs_num, 4), dtype=np.float32)
        self.dyn_true_traj = [[] for _ in range(self.DYN_MAX)]
        self.dyn_est_traj = [[] for _ in range(self.DYN_MAX)]

    def _normalize_dyn_traj(self, arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim == 2:
            return a.reshape(a.shape[0], 1, 4).astype(np.float32)
        if a.ndim == 3:
            return a.astype(np.float32)
        raise ValueError(f"Invalid dyn_traj shape {a.shape}")

    def _init_dyn_colors(self):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab10")
        num = max(int(getattr(self, "DYN_MAX", 0)), 1)
        self.dyn_colors = [cmap(i % cmap.N) for i in range(num)]

    def render(self, mode="human"):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.ax.imshow(self.grid_map, cmap="gray_r", origin="lower", alpha=1.0)

        for i in range(self.dyn_obs_num):
            row = self.dyn_pos[i]
            x, y = float(row[0]), float(row[1])
            vx, vy = float(row[2]), float(row[3])
            speed2 = vx * vx + vy * vy
            psi_obs = float(np.arctan2(vy, vx)) if speed2 > 1e-14 else (float(row[4]) if row.shape[0] > 4 else 0.0)
            color = self.dyn_colors[i] if hasattr(self, "dyn_colors") and i < len(self.dyn_colors) else "gray"
            draw_obstacles(self.ax, x, y, psi_obs, length=2.0, width=1.08, color=color)

        for i in range(self.dyn_obs_num):
            traj = [p for p in self.dyn_est_traj[i] if p is not None]
            if len(traj) > 1:
                xs, ys = zip(*traj)
                color = self.dyn_colors[i] if hasattr(self, "dyn_colors") and i < len(self.dyn_colors) else "gray"
                self.ax.plot(xs, ys, color=color, linewidth=1.5, linestyle="-", alpha=0.8)

        if len(self.path) > 0:
            x0, y0 = self.path[0]
            self.ax.plot(x0, y0, marker="s", color="blue", markersize=8, label="Start")

        x, y = self.ship_state[0], self.ship_state[1]
        hitbox = plt.Circle((x, y), self.R_usv, color="red", fill=False, linestyle="--", linewidth=1, alpha=0.5)
        self.ax.add_patch(hitbox)

        for i in range(self.dyn_obs_num):
            obs_x, obs_y = float(self.dyn_pos[i, 0]), float(self.dyn_pos[i, 1])
            obs_hitbox = plt.Circle((obs_x, obs_y), self.dyn_radius, color=self.dyn_colors[i], fill=False, linestyle="--", alpha=0.5)
            self.ax.add_patch(obs_hitbox)

        x, y, psi_ship = self.ship_state[0], self.ship_state[1], self.ship_state[2]
        draw_usv(self.ax, x, y, psi_ship, length=2.0, width=1.08, color="red")

        gx, gy = self.goal
        self.ax.plot(gx, gy, marker="*", color="blue", markersize=12, label="Goal")

        if len(self.path) > 1:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.0, label="USV trajectory")

        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(f"Step={self.current_step}")
        self.ax.legend(loc="lower right")

        if mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            rgba = np.asarray(self.fig.canvas.buffer_rgba())
            return rgba[..., :3].copy()
        else:
            raise ValueError(f"Unknown mode {mode}")
