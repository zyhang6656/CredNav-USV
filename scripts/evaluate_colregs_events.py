"""Offline event-level COLREGs evaluation for recorded formal runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from scripts.compare_safety_filters import exact_mcnemar
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from compare_safety_filters import exact_mcnemar


REQUIRED_STEP_COLUMNS = {
    "map", "seed", "step", "ship_x", "ship_y", "ship_psi",
    "ship_u", "ship_v", "ship_r", "action_yaw",
}

METHOD_LABELS = {
    "cwvl": "CWVL",
    "cbf_vo": "CBF-VO",
    "full_corecbf": "CoReCBF-QP",
    "drl_vo": "DRL-VO",
    "colregs_mpcc": "COLREGs-MPCC",
    "lag_u": "Lag-U",
}

SENSITIVITY_VALUES = {
    "dcpa_warning": (2.5, 3.0, 3.5),
    "tcpa_horizon": (8.0, 10.0, 12.0),
    "reciprocal_tolerance_deg": (10.0, 15.0, 20.0),
    "action_course_deg": (3.0, 5.0, 7.0),
    "substantial_course_deg": (10.0, 15.0, 20.0),
    "response_ratio_limit": (0.20, 0.25, 0.33),
}

EVENT_FIELDS = [
    "map", "seed", "method", "obstacles", "event_id",
    "overlapping_event_ids", "target_id", "rule", "role", "valid_primary",
    "censored", "censor_reason",
    "start_index", "end_index", "start_step", "end_step", "t_start",
    "t_action", "t_cpa", "t_end", "initial_bearing", "own_course",
    "target_course", "resolved", "action_index", "action_active_at_entry",
    "pre_existing_maneuver", "response_seconds", "response_ratio", "timely",
    "tcpa_start", "dcpa_start", "course_change", "speed_reduction",
    "max_starboard_course_change_deg", "max_port_course_change_deg",
    "max_abs_course_change_deg", "max_speed_reduction_fraction",
    "starboard_substantial", "port_substantial", "course_substantial",
    "speed_substantial", "cpa_index", "min_distance", "minimum_distance",
    "passing_side", "safe", "substantial", "compliant", "failure_reason",
]


@dataclass(frozen=True)
class EvaluationConfig:
    dt: float = 0.1
    speed_floor: float = 0.1
    tcpa_horizon: float = 10.0
    dcpa_warning: float = 3.0
    safety_distance: float = 2.0
    enter_duration: float = 0.5
    exit_duration: float = 1.0
    merge_gap_duration: float = 1.0
    head_bearing_deg: float = 5.0
    reciprocal_tolerance_deg: float = 15.0
    crossing_limit_deg: float = 112.5
    action_course_deg: float = 5.0
    action_speed_fraction: float = 0.10
    action_hold_duration: float = 0.5
    substantial_course_deg: float = 15.0
    substantial_speed_fraction: float = 0.20
    substantial_hold_duration: float = 1.0
    response_ratio_limit: float = 0.25
    response_seconds_limit: float = 2.0
    passing_window: float = 0.5
    passing_deadband: float = 0.25

    def __post_init__(self) -> None:
        values = np.asarray(
            [float(getattr(self, item.name)) for item in fields(self)],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("evaluation thresholds must be finite and positive")
        if self.safety_distance >= self.dcpa_warning:
            raise ValueError("safety distance must be below warning distance")

    def _steps(self, duration: float) -> int:
        return max(1, int(round(float(duration) / self.dt)))

    @property
    def enter_steps(self) -> int:
        return self._steps(self.enter_duration)

    @property
    def exit_steps(self) -> int:
        return self._steps(self.exit_duration)

    @property
    def merge_gap_steps(self) -> int:
        return self._steps(self.merge_gap_duration)

    @property
    def action_hold_steps(self) -> int:
        return self._steps(self.action_hold_duration)

    @property
    def substantial_hold_steps(self) -> int:
        return self._steps(self.substantial_hold_duration)


@dataclass(frozen=True)
class EpisodeSeries:
    map_name: str
    seed: int
    steps: np.ndarray
    own_position: np.ndarray
    own_course: np.ndarray
    own_speed: np.ndarray
    own_surge: np.ndarray
    own_yaw_rate: np.ndarray
    action_yaw: np.ndarray
    target_state: np.ndarray
    target_boundary_hold_samples: int = 0


@dataclass(frozen=True)
class EpisodeGeometry:
    distance: np.ndarray
    tcpa: np.ndarray
    dcpa: np.ndarray
    target_speed: np.ndarray
    distance_increasing: np.ndarray


@dataclass(frozen=True)
class OpportunityDigest:
    steps: np.ndarray
    risk: np.ndarray
    distance: np.ndarray


@dataclass
class SourceEvaluation:
    events: list[dict[str, object]]
    sensitivity_events: list[dict[str, object]]
    episode_summaries: list[dict[str, object]]
    opportunity_digests: dict[tuple[str, int, str, int], OpportunityDigest]
    input_files: list[dict[str, object]]
    map_inputs: dict[Path, str]


def wrap_pi(angle):
    wrapped = (np.asarray(angle, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def course_frame(course: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([math.cos(course), math.sin(course)], dtype=float),
        np.array([-math.sin(course), math.cos(course)], dtype=float),
    )


def signed_starboard_bearing(relative_position, own_course: float) -> float:
    forward, port = course_frame(own_course)
    relative = np.asarray(relative_position, dtype=float)
    return float(-math.atan2(float(relative @ port), float(relative @ forward)))


def tcpa_dcpa(relative_position, relative_velocity) -> tuple[float, float]:
    relative = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    speed_squared = float(velocity @ velocity)
    if speed_squared <= 1e-12:
        return 0.0, float(np.linalg.norm(relative))
    tcpa = float(-(relative @ velocity) / speed_squared)
    return tcpa, float(np.linalg.norm(relative + tcpa * velocity))


def classify_encounter(
    own_position: np.ndarray,
    own_course: float,
    own_speed: float,
    target_state: np.ndarray,
    config: EvaluationConfig,
) -> tuple[str, str] | None:
    target = np.asarray(target_state, dtype=float)
    target_position = target[:2]
    target_velocity = target[2:]
    target_speed = float(np.linalg.norm(target_velocity))
    if own_speed < config.speed_floor or target_speed < config.speed_floor:
        return None

    own_forward, _own_port = course_frame(own_course)
    own_velocity = float(own_speed) * own_forward
    relative_position = target_position - np.asarray(own_position, dtype=float)
    relative_velocity = target_velocity - own_velocity
    tcpa, dcpa = tcpa_dcpa(relative_position, relative_velocity)
    if not (0.0 < tcpa <= config.tcpa_horizon and dcpa <= config.dcpa_warning):
        return None

    target_course = math.atan2(float(target_velocity[1]), float(target_velocity[0]))
    own_bearing = signed_starboard_bearing(relative_position, own_course)
    target_bearing = signed_starboard_bearing(-relative_position, target_course)
    head_limit = math.radians(config.head_bearing_deg)
    reciprocal_limit = math.radians(config.reciprocal_tolerance_deg)
    course_delta = abs(float(wrap_pi(target_course - own_course)))
    reciprocal = abs(math.pi - course_delta) <= reciprocal_limit
    mutual_bow = abs(own_bearing) <= head_limit and abs(target_bearing) <= head_limit
    if reciprocal and mutual_bow:
        return "rule14", "give_way"

    target_forward, _target_port = course_frame(target_course)
    closing_from_astern = float((own_velocity - target_velocity) @ target_forward) > 0.0
    abaft_limit = math.radians(112.5)
    if abs(target_bearing) >= abaft_limit and closing_from_astern:
        return None

    crossing_limit = math.radians(config.crossing_limit_deg)
    if head_limit < own_bearing <= crossing_limit:
        return "rule15", "give_way"
    if -crossing_limit <= own_bearing < -head_limit:
        return "rule15", "stand_on"
    return None


def build_event_windows(
    risk: np.ndarray,
    clear: np.ndarray,
    config: EvaluationConfig,
) -> list[dict[str, int | bool]]:
    risk = np.asarray(risk, dtype=bool)
    clear = np.asarray(clear, dtype=bool)
    if risk.shape != clear.shape or risk.ndim != 1:
        raise ValueError("risk and clear masks must be one-dimensional and aligned")

    windows: list[dict[str, int | bool]] = []
    candidate_start: int | None = None
    entry_run = 0
    active_start: int | None = None
    clear_run = 0
    for index, (is_risk, is_clear) in enumerate(zip(risk, clear)):
        if active_start is None:
            if is_risk:
                candidate_start = index if candidate_start is None else candidate_start
                entry_run += 1
                if entry_run >= config.enter_steps:
                    active_start = int(candidate_start)
                    clear_run = 0
            else:
                candidate_start = None
                entry_run = 0
            continue

        clear_run = clear_run + 1 if is_clear else 0
        if clear_run >= max(config.exit_steps, config.merge_gap_steps):
            windows.append({"start": active_start, "end": index, "resolved": True})
            active_start = None
            candidate_start = None
            entry_run = 0
            clear_run = 0

    if active_start is not None:
        windows.append({
            "start": active_start,
            "end": int(risk.size - 1),
            "resolved": False,
        })
    return windows


def compute_episode_geometry(series: EpisodeSeries) -> EpisodeGeometry:
    relative_position = (
        series.target_state[:, :, :2] - series.own_position[:, None, :]
    )
    own_velocity = series.own_speed[:, None] * np.column_stack((
        np.cos(series.own_course), np.sin(series.own_course)
    ))
    relative_velocity = series.target_state[:, :, 2:] - own_velocity[:, None, :]
    speed_squared = np.sum(relative_velocity * relative_velocity, axis=2)
    projection = np.sum(relative_position * relative_velocity, axis=2)
    tcpa = np.zeros_like(speed_squared)
    moving = speed_squared > 1e-12
    tcpa[moving] = -projection[moving] / speed_squared[moving]
    distance = np.linalg.norm(relative_position, axis=2)
    return EpisodeGeometry(
        distance=distance,
        tcpa=tcpa,
        dcpa=np.linalg.norm(
            relative_position + tcpa[:, :, None] * relative_velocity, axis=2
        ),
        target_speed=np.linalg.norm(series.target_state[:, :, 2:], axis=2),
        distance_increasing=np.vstack((
            np.zeros((1, distance.shape[1]), dtype=bool),
            np.diff(distance, axis=0) > 0.0,
        )),
    )


def _target_geometry(
    series: EpisodeSeries,
    target_id: int,
    config: EvaluationConfig,
    geometry: EpisodeGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = compute_episode_geometry(series) if geometry is None else geometry
    risk = (
        (series.own_speed >= config.speed_floor)
        & (geometry.target_speed[:, target_id] >= config.speed_floor)
        & (geometry.tcpa[:, target_id] > 0.0)
        & (geometry.tcpa[:, target_id] <= config.tcpa_horizon)
        & (geometry.dcpa[:, target_id] <= config.dcpa_warning)
    )
    clear = (
        (geometry.tcpa[:, target_id] <= 0.0)
        & (geometry.distance[:, target_id] >= config.dcpa_warning)
        & geometry.distance_increasing[:, target_id]
    )
    return risk, clear


def build_opportunity_digest(
    series: EpisodeSeries,
    geometry: EpisodeGeometry,
    config: EvaluationConfig,
) -> OpportunityDigest:
    risk = (
        (series.own_speed[:, None] >= config.speed_floor)
        & (geometry.target_speed >= config.speed_floor)
        & (geometry.tcpa > 0.0)
        & (geometry.tcpa <= config.tcpa_horizon)
        & (geometry.dcpa <= config.dcpa_warning)
    )
    return OpportunityDigest(
        steps=series.steps.copy(),
        risk=risk,
        distance=geometry.distance.astype(np.float32),
    )


def classify_absent_opportunity(
    event: dict[str, object],
    digest: OpportunityDigest,
    config: EvaluationConfig,
) -> str:
    target_id = int(event["target_id"])
    if target_id >= digest.risk.shape[1]:
        return "not_evaluable"
    in_window = (
        (digest.steps >= int(event["start_step"]))
        & (digest.steps <= int(event["end_step"]))
    )
    if not np.any(in_window):
        return "not_evaluable"
    safe = bool(np.all(
        digest.distance[in_window, target_id] >= config.safety_distance
    ))
    return (
        "pre_entry_avoidance"
        if safe and not np.any(digest.risk[in_window, target_id])
        else "not_evaluable"
    )


def first_sustained(mask: np.ndarray, steps: int, stop: int) -> int | None:
    values = np.asarray(mask, dtype=bool)
    width = max(1, int(steps))
    last_start = min(int(stop), values.size - 1) - width + 1
    for start in range(max(0, last_start + 1)):
        if bool(np.all(values[start:start + width])):
            return start
    return None


def _event_cpa(series: EpisodeSeries, target_id: int, start: int, end: int) -> int:
    relative = (
        series.target_state[start:end + 1, target_id, :2]
        - series.own_position[start:end + 1]
    )
    return int(start + np.argmin(np.linalg.norm(relative, axis=1)))


def action_metrics(
    series: EpisodeSeries,
    start: int,
    cpa: int,
    config: EvaluationConfig,
    target_id: int = 0,
    baseline_index: int | None = None,
) -> dict[str, object]:
    baseline = start if baseline_index is None else int(baseline_index)
    if not 0 <= baseline <= start:
        raise ValueError("action baseline must not follow event entry")
    baseline_course = float(series.own_course[baseline])
    baseline_speed = max(float(series.own_speed[baseline]), 1e-9)
    course_change = np.asarray(wrap_pi(series.own_course - baseline_course))
    course_change_deg = np.degrees(course_change)
    speed_reduction = (baseline_speed - series.own_speed) / baseline_speed
    action_mask = (
        (np.abs(course_change_deg) >= config.action_course_deg)
        | (speed_reduction >= config.action_speed_fraction)
    )
    action_index = first_sustained(
        action_mask[baseline:cpa + 1], config.action_hold_steps, cpa - baseline
    )
    if action_index is not None:
        action_index += baseline

    target = series.target_state[start, target_id]
    own_forward, _own_port = course_frame(float(series.own_course[start]))
    own_velocity = float(series.own_speed[start]) * own_forward
    tcpa_start, dcpa_start = tcpa_dcpa(
        target[:2] - series.own_position[start],
        target[2:] - own_velocity,
    )
    if action_index is None:
        response_seconds = math.inf
        response_ratio = math.inf
    else:
        response_seconds = float(max(0, action_index - start) * config.dt)
        response_ratio = (
            response_seconds / tcpa_start if tcpa_start > 1e-12 else math.inf
        )
    timely = bool(
        action_index is not None
        and response_seconds <= config.response_seconds_limit
        and response_ratio <= config.response_ratio_limit
    )

    interval = slice(baseline, cpa + 1)
    interval_stop = cpa - baseline
    starboard_deg = -course_change_deg
    starboard_substantial = first_sustained(
        starboard_deg[interval] >= config.substantial_course_deg,
        config.substantial_hold_steps,
        interval_stop,
    ) is not None
    port_substantial = first_sustained(
        course_change_deg[interval] >= config.substantial_course_deg,
        config.substantial_hold_steps,
        interval_stop,
    ) is not None
    course_substantial = first_sustained(
        np.abs(course_change_deg[interval]) >= config.substantial_course_deg,
        config.substantial_hold_steps,
        interval_stop,
    ) is not None
    speed_substantial = first_sustained(
        speed_reduction[interval] >= config.substantial_speed_fraction,
        config.substantial_hold_steps,
        interval_stop,
    ) is not None
    return {
        "action_index": action_index,
        "response_seconds": response_seconds,
        "response_ratio": response_ratio,
        "timely": timely,
        "action_active_at_entry": bool(
            action_index is not None
            and action_index <= start
            and action_mask[start]
        ),
        "tcpa_start": float(tcpa_start),
        "dcpa_start": float(dcpa_start),
        "max_starboard_course_change_deg": float(np.max(starboard_deg[interval])),
        "max_port_course_change_deg": float(np.max(course_change_deg[interval])),
        "max_abs_course_change_deg": float(np.max(np.abs(course_change_deg[interval]))),
        "max_speed_reduction_fraction": float(np.max(speed_reduction[interval])),
        "starboard_substantial": bool(starboard_substantial),
        "port_substantial": bool(port_substantial),
        "course_substantial": bool(course_substantial),
        "speed_substantial": bool(speed_substantial),
    }


def rule14_passing_side(
    series: EpisodeSeries,
    target_id: int,
    cpa: int,
    config: EvaluationConfig,
) -> str:
    radius = config._steps(config.passing_window)
    start = max(0, cpa - radius)
    end = min(series.steps.size, cpa + radius + 1)
    own_projection = []
    target_projection = []
    for index in range(start, end):
        relative = (
            series.target_state[index, target_id, :2]
            - series.own_position[index]
        )
        _own_forward, own_port = course_frame(float(series.own_course[index]))
        target_velocity = series.target_state[index, target_id, 2:]
        target_course = math.atan2(
            float(target_velocity[1]), float(target_velocity[0])
        )
        _target_forward, target_port = course_frame(target_course)
        own_projection.append(float(relative @ own_port))
        target_projection.append(float((-relative) @ target_port))
    own_side = float(np.median(own_projection))
    target_side = float(np.median(target_projection))
    margin = config.passing_deadband
    if own_side > margin and target_side > margin:
        return "port_to_port"
    if own_side < -margin and target_side < -margin:
        return "starboard_to_starboard"
    return "ambiguous"


def rule15_passing_side(
    series: EpisodeSeries,
    target_id: int,
    start: int,
    end: int,
    config: EvaluationConfig,
) -> str:
    target_velocity = series.target_state[start, target_id, 2:]
    target_course = math.atan2(
        float(target_velocity[1]), float(target_velocity[0])
    )
    target_forward, target_port = course_frame(target_course)
    relative = (
        series.own_position[start:end + 1]
        - series.target_state[start:end + 1, target_id, :2]
    )
    lateral = relative @ target_port
    longitudinal = relative @ target_forward
    outcomes: list[str] = []
    for index in range(lateral.size - 1):
        left = float(lateral[index])
        right = float(lateral[index + 1])
        if left == 0.0 or left * right > 0.0:
            continue
        fraction = left / (left - right) if left != right else 0.0
        along = float(
            longitudinal[index]
            + fraction * (longitudinal[index + 1] - longitudinal[index])
        )
        if along > config.passing_deadband:
            outcomes.append("ahead")
        elif along < -config.passing_deadband:
            outcomes.append("astern")
        else:
            outcomes.append("ambiguous")
    if "ahead" in outcomes:
        return "ahead"
    if "ambiguous" in outcomes:
        return "ambiguous"
    if outcomes:
        return "astern"
    return "no_crossing_clear"


FAILURE_PRECEDENCE = (
    "unsafe_clearance",
    "unresolved",
    "port_turn_head_on",
    "wrong_passing_side",
    "crossed_ahead",
    "ambiguous_passing",
    "late_action",
    "insufficient_action",
)


def score_event(
    series: EpisodeSeries,
    event: dict[str, object],
    config: EvaluationConfig,
) -> dict[str, object]:
    scored = dict(event)
    start = int(event["start_index"])
    end = int(event["end_index"])
    target_id = int(event["target_id"])
    cpa = int(event.get("cpa_index", _event_cpa(series, target_id, start, end)))
    metrics = action_metrics(series, start, cpa, config, target_id)
    relative = (
        series.target_state[start:end + 1, target_id, :2]
        - series.own_position[start:end + 1]
    )
    minimum_distance = float(np.min(np.linalg.norm(relative, axis=1)))
    safe = bool(minimum_distance >= config.safety_distance)
    resolved = bool(event["resolved"])
    rule = str(event["rule"])
    if rule == "rule14":
        passing_side = rule14_passing_side(series, target_id, cpa, config)
        passing_valid = passing_side == "port_to_port"
    else:
        passing_side = rule15_passing_side(series, target_id, start, end, config)
        passing_valid = passing_side in {"astern", "no_crossing_clear"}

    lookback = max(0, start - config._steps(config.response_seconds_limit))
    prior_metrics = action_metrics(
        series, start, cpa, config, target_id, baseline_index=lookback
    )
    prior_substantial = bool(
        prior_metrics["starboard_substantial"]
        if rule == "rule14"
        else prior_metrics["course_substantial"]
        or prior_metrics["speed_substantial"]
    )
    pre_existing = bool(
        lookback < start
        and prior_metrics["action_active_at_entry"]
        and prior_substantial
        and passing_valid
    )
    if pre_existing:
        metrics = prior_metrics
        metrics.update({
            "action_index": start,
            "response_seconds": 0.0,
            "response_ratio": 0.0,
            "timely": True,
        })
    metrics["pre_existing_maneuver"] = pre_existing

    if rule == "rule14":
        substantial = bool(metrics["starboard_substantial"])
    else:
        substantial = bool(
            metrics["course_substantial"] or metrics["speed_substantial"]
        )

    failures = {
        "unsafe_clearance": not safe,
        "unresolved": not resolved,
        "port_turn_head_on": bool(
            rule == "rule14"
            and metrics["port_substantial"]
            and not metrics["starboard_substantial"]
        ),
        "wrong_passing_side": bool(
            rule == "rule14" and passing_side == "starboard_to_starboard"
        ),
        "crossed_ahead": bool(rule == "rule15" and passing_side == "ahead"),
        "ambiguous_passing": bool(passing_side == "ambiguous"),
        "late_action": not bool(metrics["timely"]),
        "insufficient_action": not substantial,
    }
    failure_reason = next(
        (name for name in FAILURE_PRECEDENCE if failures[name]), ""
    )
    compliant = bool(
        bool(metrics["timely"])
        and substantial
        and safe
        and resolved
        and (
            passing_valid
        )
    )
    scored.update(metrics)
    scored.update({
        "cpa_index": cpa,
        "t_action": (
            float(series.steps[int(metrics["action_index"])] * config.dt)
            if metrics["action_index"] is not None else None
        ),
        "course_change": (
            float(metrics["max_starboard_course_change_deg"])
            if rule == "rule14"
            else float(metrics["max_abs_course_change_deg"])
        ),
        "speed_reduction": float(metrics["max_speed_reduction_fraction"]),
        "min_distance": minimum_distance,
        "minimum_distance": minimum_distance,
        "passing_side": passing_side,
        "safe": safe,
        "substantial": substantial,
        "compliant": compliant,
        "failure_reason": failure_reason,
    })
    return scored


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def episode_control_quality(
    series: EpisodeSeries,
    config: EvaluationConfig,
) -> dict[str, float]:
    low_speed = series.own_speed < 0.05
    reverse = series.own_surge < -0.05
    return {
        "longest_low_speed_s": float(_longest_run(low_speed) * config.dt),
        "longest_reverse_s": float(_longest_run(reverse) * config.dt),
        "reverse_distance_m": float(
            np.sum(np.maximum(-series.own_surge, 0.0)) * config.dt
        ),
        "yaw_action_total_variation": float(
            np.sum(np.abs(np.diff(series.action_yaw)))
        ),
        "max_abs_yaw_rate": float(np.max(np.abs(series.own_yaw_rate))),
    }


def episode_series_from_rows(
    rows: list[dict[str, str]],
    map_path: Path,
) -> EpisodeSeries:
    if not rows:
        raise ValueError("episode step rows must not be empty")
    missing = REQUIRED_STEP_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"step rows are missing required columns: {sorted(missing)}")
    map_name = str(rows[0]["map"])
    seed = int(rows[0]["seed"])
    if any((str(row["map"]), int(row["seed"])) != (map_name, seed) for row in rows):
        raise ValueError("episode step rows must share one map and seed")

    steps = np.asarray([int(row["step"]) for row in rows], dtype=int)
    if np.any(np.diff(steps) != 1):
        raise ValueError("episode step indices must be contiguous and increasing")
    with np.load(map_path, allow_pickle=False) as data:
        if "dyn_traj" not in data:
            raise ValueError(f"scenario has no dyn_traj: {map_path}")
        dynamic = np.asarray(data["dyn_traj"], dtype=float)
    if dynamic.ndim != 3 or dynamic.shape[0] == 0 or dynamic.shape[2] != 4:
        raise ValueError("dyn_traj must have shape [time, obstacle, 4]")
    if steps[0] < 0:
        raise ValueError("recorded episode steps must be nonnegative")
    target_indices = np.minimum(steps, dynamic.shape[0] - 1)

    psi = np.asarray([float(row["ship_psi"]) for row in rows], dtype=float)
    surge = np.asarray([float(row["ship_u"]) for row in rows], dtype=float)
    sway = np.asarray([float(row["ship_v"]) for row in rows], dtype=float)
    vx = surge * np.cos(psi) - sway * np.sin(psi)
    vy = surge * np.sin(psi) + sway * np.cos(psi)
    speed = np.hypot(vx, vy)
    course = np.where(speed >= 1e-9, np.arctan2(vy, vx), psi)
    return EpisodeSeries(
        map_name=map_name,
        seed=seed,
        steps=steps,
        own_position=np.asarray([
            [float(row["ship_x"]), float(row["ship_y"])] for row in rows
        ]),
        own_course=course,
        own_speed=speed,
        own_surge=surge,
        own_yaw_rate=np.asarray([float(row["ship_r"]) for row in rows]),
        action_yaw=np.asarray([float(row["action_yaw"]) for row in rows]),
        target_state=dynamic[target_indices],
        target_boundary_hold_samples=int(np.count_nonzero(steps >= dynamic.shape[0])),
    )


def episode_summary_row(
    series: EpisodeSeries,
    events: list[dict[str, object]],
    *,
    method: str,
    obstacles: int,
    config: EvaluationConfig,
) -> dict[str, object]:
    primary = [event for event in events if bool(event.get("valid_primary", False))]
    compliant = sum(bool(event.get("compliant", False)) for event in primary)
    count = len(primary)
    row: dict[str, object] = {
        "map": series.map_name,
        "seed": int(series.seed),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "obstacles": int(obstacles),
        "valid_event_count": count,
        "compliant_event_count": compliant,
        "event_compliance": float(compliant / count) if count else None,
        "strict_episode_compliance": bool(compliant == count) if count else None,
        "target_boundary_hold_samples": int(series.target_boundary_hold_samples),
    }
    row.update(episode_control_quality(series, config))
    return row


def build_sensitivity_configs(
    base: EvaluationConfig,
) -> list[tuple[str, EvaluationConfig]]:
    profiles = [("primary", base)]
    for name, values in SENSITIVITY_VALUES.items():
        for value in values:
            if math.isclose(float(getattr(base, name)), value):
                continue
            profiles.append((f"{name}_{value:g}", replace(base, **{name: value})))
    profiles.extend([
        (
            "lenient_action",
            replace(
                base,
                action_course_deg=3.0,
                substantial_course_deg=10.0,
                response_ratio_limit=0.33,
            ),
        ),
        (
            "strict_action",
            replace(
                base,
                action_course_deg=7.0,
                substantial_course_deg=20.0,
                response_ratio_limit=0.20,
            ),
        ),
    ])
    return profiles


def match_materialized_events(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    left = sorted(
        (row for row in left if bool(row.get("valid_primary", False))),
        key=lambda row: (
            str(row["map"]), int(row["seed"]), int(row["target_id"]),
            int(row["start_step"]),
        ),
    )
    right = sorted(
        (row for row in right if bool(row.get("valid_primary", False))),
        key=lambda row: (
            str(row["map"]), int(row["seed"]), int(row["target_id"]),
            int(row["start_step"]),
        ),
    )
    used: set[int] = set()
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for left_row in left:
        for index, right_row in enumerate(right):
            if index in used:
                continue
            keys = ("map", "seed", "obstacles", "target_id")
            if any(left_row.get(key) != right_row.get(key) for key in keys):
                continue
            if max(int(left_row["start_step"]), int(right_row["start_step"])) > min(
                int(left_row["end_step"]), int(right_row["end_step"])
            ):
                continue
            used.add(index)
            pairs.append((left_row, right_row))
            break
    return pairs


def _passing_success(event: dict[str, object]) -> bool:
    if event["rule"] == "rule14":
        return event.get("passing_side") == "port_to_port"
    return event.get("passing_side") in {"astern", "no_crossing_clear"}


def summarize_events(
    events: list[dict[str, object]],
    episodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    primary = [row for row in events if bool(row.get("valid_primary", False))]
    groups = sorted({
        (str(row["method"]), int(row["obstacles"]))
        for row in [*primary, *episodes]
    })
    rows: list[dict[str, object]] = []
    rate_lookup: dict[tuple[str, int, str], float | None] = {}
    strict_lookup: dict[tuple[str, int], float | None] = {}
    components = ("timely", "substantial", "safe", "resolved", "passing")
    for method, count in groups:
        group_events = [
            row for row in primary
            if row["method"] == method and int(row["obstacles"]) == count
        ]
        for rule in ("combined", "rule14", "rule15"):
            selected = (
                group_events if rule == "combined"
                else [row for row in group_events if row["rule"] == rule]
            )
            rate = (
                float(np.mean([bool(row["compliant"]) for row in selected]))
                if selected else None
            )
            rate_lookup[(method, count, rule)] = rate
            rows.append({
                "record_type": "event_rate",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "obstacles": count,
                "rule": rule,
                "valid_event_count": len(selected),
                "compliant_event_count": sum(
                    bool(row["compliant"]) for row in selected
                ),
                "rate": rate,
            })
            for component in components:
                component_rows = [
                    {
                        **row,
                        "compliant": (
                            _passing_success(row)
                            if component == "passing" else bool(row[component])
                        ),
                    }
                    for row in selected
                ]
                component_rate = (
                    float(np.mean([row["compliant"] for row in component_rows]))
                    if component_rows else None
                )
                rows.append({
                    "record_type": "component_rate",
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "obstacles": count,
                    "rule": rule,
                    "component": component,
                    "valid_event_count": len(component_rows),
                    "compliant_event_count": sum(
                        bool(row["compliant"]) for row in component_rows
                    ),
                    "rate": component_rate,
                })

        group_episodes = [
            row for row in episodes
            if row["method"] == method
            and int(row["obstacles"]) == count
            and row.get("strict_episode_compliance") is not None
        ]
        strict_rate = (
            float(np.mean([
                bool(row["strict_episode_compliance"]) for row in group_episodes
            ])) if group_episodes else None
        )
        strict_lookup[(method, count)] = strict_rate
        rows.append({
            "record_type": "strict_episode_rate",
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "obstacles": count,
            "rule": "combined",
            "valid_episode_count": len(group_episodes),
            "compliant_episode_count": sum(
                bool(row["strict_episode_compliance"]) for row in group_episodes
            ),
            "rate": strict_rate,
        })

    for method in sorted({method for method, _count in groups}):
        for rule in ("combined", "rule14", "rule15"):
            density_rates = [
                rate_lookup[(method, count, rule)]
                for group_method, count in groups if group_method == method
                if rate_lookup[(method, count, rule)] is not None
            ]
            rows.append({
                "record_type": "macro_event_rate",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "obstacles": "all",
                "rule": rule,
                "density_count": len(density_rates),
                "rate": float(np.mean(density_rates)) if density_rates else None,
            })
        strict_rates = [
            strict_lookup[(method, count)]
            for group_method, count in groups if group_method == method
            if strict_lookup[(method, count)] is not None
        ]
        rows.append({
            "record_type": "macro_strict_episode_rate",
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "obstacles": "all",
            "rule": "combined",
            "density_count": len(strict_rates),
            "rate": float(np.mean(strict_rates)) if strict_rates else None,
        })

    paired_rows: list[dict[str, object]] = []
    for comparator in ("cwvl", "cbf_vo"):
        for count in sorted({count for _method, count in groups}):
            for rule in ("combined", "rule14", "rule15"):
                left = [
                    row for row in primary
                    if row["method"] == comparator
                    and int(row["obstacles"]) == count
                    and (rule == "combined" or row["rule"] == rule)
                ]
                right = [
                    row for row in primary
                    if row["method"] == "full_corecbf"
                    and int(row["obstacles"]) == count
                    and (rule == "combined" or row["rule"] == rule)
                ]
                pairs = match_materialized_events(left, right)
                if not pairs:
                    continue
                left_values = [bool(left_row["compliant"]) for left_row, _ in pairs]
                right_values = [bool(right_row["compliant"]) for _, right_row in pairs]
                test = exact_mcnemar(left_values, right_values)
                paired_rows.append({
                    "record_type": "paired_difference",
                    "method": "full_corecbf",
                    "method_label": METHOD_LABELS["full_corecbf"],
                    "comparator": comparator,
                    "obstacles": count,
                    "rule": rule,
                    "paired_event_count": len(pairs),
                    "rate": float(np.mean(right_values) - np.mean(left_values)),
                    "comparator_only_success": test["left_only_success"],
                    "core_only_success": test["right_only_success"],
                    "p_value": test["p_value_two_sided"],
                })
    rows.extend(paired_rows)
    return rows


def summarize_unmatched_opportunities(
    events: list[dict[str, object]],
    digests: dict[tuple[str, int, str, int], OpportunityDigest],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    primary = [row for row in events if bool(row.get("valid_primary", False))]
    counts: dict[tuple[str, str, int, str, str], int] = {}
    for comparator in ("cwvl", "cbf_vo"):
        left = [row for row in primary if row["method"] == comparator]
        right = [row for row in primary if row["method"] == "full_corecbf"]
        pairs = match_materialized_events(left, right)
        matched_left = {id(row) for row, _other in pairs}
        matched_right = {id(row) for _other, row in pairs}
        for reference, other_method, matched in (
            (left, "full_corecbf", matched_left),
            (right, comparator, matched_right),
        ):
            for event in reference:
                if id(event) in matched:
                    continue
                digest = digests.get((
                    other_method,
                    int(event["obstacles"]),
                    str(event["map"]),
                    int(event["seed"]),
                ))
                status = (
                    classify_absent_opportunity(event, digest, config)
                    if digest is not None else "not_evaluable"
                )
                key = (
                    str(event["method"]), other_method,
                    int(event["obstacles"]), str(event["rule"]), status,
                )
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "record_type": "opportunity_accounting",
            "reference_method": reference,
            "other_method": other,
            "obstacles": obstacles,
            "rule": rule,
            "status": status,
            "event_count": count,
        }
        for (reference, other, obstacles, rule, status), count in sorted(counts.items())
    ]


def iter_episode_rows(
    path: Path,
) -> Iterator[tuple[tuple[str, int], list[dict[str, str]]]]:
    seen: set[tuple[str, int]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_STEP_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"step CSV is missing required columns: {sorted(missing)}")
        current_key: tuple[str, int] | None = None
        current_rows: list[dict[str, str]] = []
        for row in reader:
            key = (str(row["map"]), int(row["seed"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                if key in seen:
                    raise ValueError(f"episode key is noncontiguous: {key}")
                seen.add(current_key)
                yield current_key, current_rows
                current_key = key
                current_rows = []
            current_rows.append(row)
        if current_key is not None:
            if current_key in seen:
                raise ValueError(f"episode key is noncontiguous: {current_key}")
            yield current_key, current_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_protocol_sources(
    report_path: Path,
    repo_root: Path,
    counts: tuple[int, ...],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    prefixes = report["protocol"]["source_prefixes"]
    sources: list[dict[str, object]] = []
    for count in counts:
        if str(count) not in prefixes:
            raise ValueError(f"summary report has no obstacle count {count}")
        for method in methods:
            if method not in METHOD_LABELS:
                raise ValueError(f"unsupported method: {method}")
            prefix = Path(prefixes[str(count)][method])
            if not prefix.is_absolute():
                prefix = Path(repo_root) / prefix
            paths = {
                "episodes_path": Path(f"{prefix}_episodes.csv"),
                "steps_path": Path(f"{prefix}_steps.csv"),
                "resolved_path": Path(f"{prefix}_resolved_config.json"),
            }
            missing = [path for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"formal source is incomplete: {missing}")
            resolved = json.loads(paths["resolved_path"].read_text(encoding="utf-8"))
            resolved_dt = resolved.get("env_kwargs", {}).get(
                "dt", resolved.get("dt")
            )
            if resolved_dt is None or not math.isclose(
                float(resolved_dt), EvaluationConfig().dt,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError(
                    f"formal source dt must equal {EvaluationConfig().dt}: "
                    f"{paths['resolved_path']}"
                )
            scenario_dirs: list[Path] = []
            for value in (
                resolved.get("scenario_dir"),
                resolved.get("env_kwargs", {}).get("scenario_dir"),
            ):
                if not value:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = Path(repo_root) / candidate
                scenario_dirs.append(candidate)
            sources.append({
                "count": int(count),
                "method": method,
                "prefix": prefix,
                "scenario_dirs": scenario_dirs,
                "resolved_dt": float(resolved_dt),
                **paths,
            })
    return sources


def _resolve_map_path(source: dict[str, object], map_name: str) -> Path:
    for directory in source["scenario_dirs"]:
        candidate = Path(directory) / map_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"scenario map is unavailable: {map_name}")


def evaluate_protocol_source(
    source: dict[str, object],
    *,
    config: EvaluationConfig,
    sensitivity_profiles: list[tuple[str, EvaluationConfig]],
    map_filter: str | None,
    seed_filter: int | None,
) -> SourceEvaluation:
    all_events: list[dict[str, object]] = []
    all_sensitivity_events: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    opportunity_digests: dict[
        tuple[str, int, str, int], OpportunityDigest
    ] = {}
    map_inputs: dict[Path, str] = {}
    episode_rows = _read_csv(source["episodes_path"])
    episode_by_key = {
        (str(row["map"]), int(row["seed"])): row for row in episode_rows
    }
    expected = set(episode_by_key)
    seen: set[tuple[str, int]] = set()
    for key, rows in iter_episode_rows(source["steps_path"]):
        if key not in expected:
            raise ValueError(f"step episode is absent from episode CSV: {key}")
        if map_filter is not None and key[0] != map_filter:
            continue
        if seed_filter is not None and key[1] != seed_filter:
            continue
        seen.add(key)
        map_path = _resolve_map_path(source, key[0]).resolve()
        if map_path not in map_inputs:
            map_inputs[map_path] = _sha256(map_path)
        series = episode_series_from_rows(rows, map_path)
        geometry = compute_episode_geometry(series)
        opportunity_digests[(
            str(source["method"]), int(source["count"]), key[0], key[1]
        )] = build_opportunity_digest(series, geometry, config)
        events: list[dict[str, object]] = []
        for profile, profile_config in sensitivity_profiles:
            profile_events = evaluate_episode(
                series,
                method=str(source["method"]),
                obstacles=int(source["count"]),
                config=profile_config,
                geometry=geometry,
                terminal_reason=str(episode_by_key[key].get("reason", "")),
            )
            all_sensitivity_events.extend(
                {**event, "profile": profile} for event in profile_events
            )
            if profile == "primary":
                events = profile_events
        all_events.extend(events)
        episode_summaries.append(episode_summary_row(
            series,
            events,
            method=str(source["method"]),
            obstacles=int(source["count"]),
            config=config,
        ))
    wanted = {
        key for key in expected
        if (map_filter is None or key[0] == map_filter)
        and (seed_filter is None or key[1] == seed_filter)
    }
    if seen != wanted:
        raise ValueError(f"episode/step key mismatch: missing={sorted(wanted - seen)}")
    input_files = [{
        "method": source["method"],
        "obstacles": source["count"],
        "episodes": str(source["episodes_path"]),
        "episodes_sha256": _sha256(source["episodes_path"]),
        "steps": str(source["steps_path"]),
        "steps_sha256": _sha256(source["steps_path"]),
        "resolved": str(source["resolved_path"]),
        "resolved_sha256": _sha256(source["resolved_path"]),
    }]
    return SourceEvaluation(
        events=all_events,
        sensitivity_events=all_sensitivity_events,
        episode_summaries=episode_summaries,
        opportunity_digests=opportunity_digests,
        input_files=input_files,
        map_inputs=map_inputs,
    )


def _source_fingerprint(
    source: dict[str, object],
    *,
    config: EvaluationConfig,
    sensitivity_profiles: list[tuple[str, EvaluationConfig]],
    map_filter: str | None,
    seed_filter: int | None,
) -> tuple[str, dict[str, object]]:
    map_inputs: list[dict[str, str]] = []
    seen_maps: set[Path] = set()
    for row in _read_csv(source["episodes_path"]):
        if map_filter is not None and row["map"] != map_filter:
            continue
        if seed_filter is not None and int(row["seed"]) != seed_filter:
            continue
        path = _resolve_map_path(source, row["map"]).resolve()
        if path in seen_maps:
            continue
        seen_maps.add(path)
        map_inputs.append({"path": str(path), "sha256": _sha256(path)})
    payload: dict[str, object] = {
        "schema_version": 1,
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "method": str(source["method"]),
        "count": int(source["count"]),
        "map_filter": map_filter,
        "seed_filter": seed_filter,
        "config": asdict(config),
        "sensitivity_profiles": [
            {"name": name, "config": asdict(profile_config)}
            for name, profile_config in sensitivity_profiles
        ],
        "inputs": {
            name: {
                "path": str(source[f"{name}_path"]),
                "sha256": _sha256(Path(source[f"{name}_path"])),
            }
            for name in ("episodes", "steps", "resolved")
        },
        "maps": map_inputs,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _source_shard_paths(
    work_dir: Path, index: int, source: dict[str, object]
) -> tuple[Path, Path]:
    stem = (
        f"source_{index:02d}_{source['method']}_obs{int(source['count'])}"
    )
    return Path(work_dir) / f"{stem}.pkl", Path(work_dir) / f"{stem}.json"


def _valid_source_shard(data_path: Path, sidecar_path: Path, fingerprint: str) -> bool:
    if not data_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        sidecar.get("schema_version") == 1
        and sidecar.get("fingerprint") == fingerprint
    )


def _replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(10):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def _write_source_shard(
    *,
    index: int,
    source: dict[str, object],
    config: EvaluationConfig,
    sensitivity_profiles: list[tuple[str, EvaluationConfig]],
    map_filter: str | None,
    seed_filter: int | None,
    work_dir: Path,
    fingerprint: str,
    fingerprint_payload: dict[str, object],
) -> tuple[Path, Path]:
    result = evaluate_protocol_source(
        source,
        config=config,
        sensitivity_profiles=sensitivity_profiles,
        map_filter=map_filter,
        seed_filter=seed_filter,
    )
    data_path, sidecar_path = _source_shard_paths(work_dir, index, source)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_temp = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    sidecar_temp = sidecar_path.with_name(f".{sidecar_path.name}.{os.getpid()}.tmp")
    with data_temp.open("wb") as handle:
        pickle.dump(result, handle, protocol=5)
    sidecar_temp.write_text(json.dumps({
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "method": str(source["method"]),
        "obstacles": int(source["count"]),
        "episodes": len(result.episode_summaries),
        "events": len(result.events),
        "worker_pid": os.getpid(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar_path.unlink(missing_ok=True)
    _replace_with_retry(data_temp, data_path)
    _replace_with_retry(sidecar_temp, sidecar_path)
    return data_path, sidecar_path


def _load_source_shard(path: Path) -> SourceEvaluation:
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, SourceEvaluation):
        raise TypeError(f"invalid COLREGs source shard: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path),
         "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _annotate_overlaps(events: list[dict[str, object]]) -> None:
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"E{index:03d}"
    for event in events:
        overlaps = [
            str(other["event_id"])
            for other in events
            if other is not event
            and max(int(event["start_step"]), int(other["start_step"]))
            <= min(int(event["end_step"]), int(other["end_step"]))
        ]
        event["overlapping_event_ids"] = ";".join(overlaps)


def _event_geometry_fields(
    series: EpisodeSeries,
    target_id: int,
    start: int,
    end: int,
    config: EvaluationConfig,
) -> dict[str, object]:
    target = series.target_state[start, target_id]
    relative_position = target[:2] - series.own_position[start]
    target_course = math.atan2(float(target[3]), float(target[2]))
    cpa = _event_cpa(series, target_id, start, end)
    distance = np.linalg.norm(
        series.target_state[start:end + 1, target_id, :2]
        - series.own_position[start:end + 1],
        axis=1,
    )
    return {
        "t_start": float(series.steps[start] * config.dt),
        "t_action": None,
        "t_cpa": float(series.steps[cpa] * config.dt),
        "t_end": float(series.steps[end] * config.dt),
        "initial_bearing": float(np.degrees(signed_starboard_bearing(
            relative_position, float(series.own_course[start])
        ))),
        "own_course": float(np.degrees(wrap_pi(series.own_course[start]))),
        "target_course": float(np.degrees(wrap_pi(target_course))),
        "cpa_index": cpa,
        "min_distance": float(np.min(distance)),
        "minimum_distance": float(np.min(distance)),
    }


def evaluate_episode(
    series: EpisodeSeries,
    method: str,
    obstacles: int,
    config: EvaluationConfig | None = None,
    geometry: EpisodeGeometry | None = None,
    terminal_reason: str = "",
) -> list[dict[str, object]]:
    config = EvaluationConfig() if config is None else config
    if series.target_state.ndim != 3 or series.target_state.shape[2] != 4:
        raise ValueError("target_state must have shape [time, obstacle, 4]")
    if series.target_state.shape[0] != series.steps.size:
        raise ValueError("episode arrays must share one time dimension")
    if int(obstacles) != series.target_state.shape[1]:
        raise ValueError("declared obstacle count does not match target_state")
    geometry = compute_episode_geometry(series) if geometry is None else geometry

    events: list[dict[str, object]] = []
    for target_id in range(int(obstacles)):
        risk, clear = _target_geometry(
            series, target_id, config, geometry=geometry
        )
        for window in build_event_windows(risk, clear, config):
            start = int(window["start"])
            censored = bool(
                not window["resolved"] and terminal_reason == "goal_reached"
            )
            classification = classify_encounter(
                series.own_position[start],
                float(series.own_course[start]),
                float(series.own_speed[start]),
                series.target_state[start, target_id],
                config,
            )
            if classification is None:
                continue
            rule, role = classification
            events.append({
                "map": series.map_name,
                "seed": int(series.seed),
                "method": str(method),
                "obstacles": int(obstacles),
                "target_id": int(target_id),
                "rule": rule,
                "role": role,
                "valid_primary": bool(role == "give_way" and not censored),
                "censored": censored,
                "censor_reason": terminal_reason if censored else "",
                "start_index": start,
                "end_index": int(window["end"]),
                "start_step": int(series.steps[start]),
                "end_step": int(series.steps[int(window["end"])]),
                "resolved": bool(window["resolved"]),
                **_event_geometry_fields(
                    series, target_id, start, int(window["end"]), config
                ),
            })
    scored = []
    for event in events:
        if bool(event["valid_primary"]):
            scored.append(score_event(series, event, config))
        elif bool(event["censored"]):
            scored.append({**event, "compliant": None, "failure_reason": ""})
        else:
            scored.append(event)
    _annotate_overlaps(scored)
    return scored


def _csv_fields(rows: list[dict[str, object]], fallback: list[str]) -> list[str]:
    result = list(fallback)
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def _write_csv(path: Path, rows: list[dict[str, object]], fallback: list[str]) -> None:
    fieldnames = _csv_fields(rows, fallback)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_summary_rows(
    events: list[dict[str, object]],
    episodes: list[dict[str, object]],
) -> list[dict[str, object]]:
    return summarize_events(events, episodes)


def _sensitivity_summary_rows(
    events: list[dict[str, object]],
    profiles: list[tuple[str, EvaluationConfig]],
    groups: list[tuple[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile, _config in profiles:
        for method, count in groups:
            selected_group = [
                row for row in events
                if row["profile"] == profile
                and row["method"] == method
                and int(row["obstacles"]) == count
                and bool(row.get("valid_primary", False))
            ]
            for rule in ("combined", "rule14", "rule15"):
                selected = (
                    selected_group if rule == "combined"
                    else [row for row in selected_group if row["rule"] == rule]
                )
                rows.append({
                    "record_type": "density",
                    "profile": profile,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "obstacles": count,
                    "rule": rule,
                    "valid_event_count": len(selected),
                    "compliant_event_count": sum(
                        bool(row["compliant"]) for row in selected
                    ),
                    "rate": (
                        float(np.mean([bool(row["compliant"]) for row in selected]))
                        if selected else None
                    ),
                })
        for method in sorted({method for method, _count in groups}):
            for rule in ("combined", "rule14", "rule15"):
                rates = [
                    row["rate"] for row in rows
                    if row["record_type"] == "density"
                    and row["profile"] == profile
                    and row["method"] == method
                    and row["rule"] == rule
                    and row["rate"] is not None
                ]
                rows.append({
                    "record_type": "macro",
                    "profile": profile,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "obstacles": "all",
                    "rule": rule,
                    "density_count": len(rates),
                    "rate": float(np.mean(rates)) if rates else None,
                })
    return rows


def _write_bundle(
    output_dir: Path,
    events: list[dict[str, object]],
    episodes: list[dict[str, object]],
    summaries: list[dict[str, object]],
    sensitivity: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporary = {
        name: output_dir / f".{name}.{token}.tmp"
        for name in (
            "colregs_events.csv",
            "colregs_episode_summary.csv",
            "colregs_summary.csv",
            "colregs_sensitivity.csv",
            "colregs_manifest.json",
        )
    }
    try:
        _write_csv(temporary["colregs_events.csv"], events, EVENT_FIELDS)
        _write_csv(
            temporary["colregs_episode_summary.csv"], episodes,
            ["map", "seed", "method", "event_compliance"],
        )
        _write_csv(
            temporary["colregs_summary.csv"], summaries,
            ["record_type", "method", "obstacles", "rate"],
        )
        _write_csv(
            temporary["colregs_sensitivity.csv"], sensitivity,
            ["profile", "method", "obstacles", "rate"],
        )
        temporary["colregs_manifest.json"].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for name, source in temporary.items():
            _replace_with_retry(source, output_dir / name)
    finally:
        for source in temporary.values():
            source.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--summary-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(7, 8, 9, 10))
    parser.add_argument(
        "--methods", nargs="+", default=("cwvl", "cbf_vo", "full_corecbf")
    )
    parser.add_argument("--map", dest="map_filter", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--no-resume-shards", action="store_true")
    args = parser.parse_args(argv)
    if int(args.workers) < 1:
        parser.error("--workers must be positive")

    repo_root = args.repo_root.resolve()
    report_path = args.summary_report
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    report_path = report_path.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = output_dir.parent / f".{output_dir.name}-work"
    elif not work_dir.is_absolute():
        work_dir = repo_root / work_dir
    work_dir = work_dir.resolve()

    config = EvaluationConfig()
    sensitivity_profiles = build_sensitivity_configs(config)
    sources = load_protocol_sources(
        report_path, repo_root, tuple(args.counts), tuple(args.methods),
    )
    all_events: list[dict[str, object]] = []
    all_sensitivity_events: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    opportunity_digests: dict[
        tuple[str, int, str, int], OpportunityDigest
    ] = {}
    input_files: list[dict[str, object]] = []
    map_inputs: dict[Path, str] = {}
    shard_paths: list[Path] = []
    pending: list[dict[str, object]] = []
    completed = 0
    for index, source in enumerate(sources):
        fingerprint, fingerprint_payload = _source_fingerprint(
            source,
            config=config,
            sensitivity_profiles=sensitivity_profiles,
            map_filter=args.map_filter,
            seed_filter=args.seed,
        )
        data_path, sidecar_path = _source_shard_paths(work_dir, index, source)
        reused = (
            not args.no_resume_shards
            and _valid_source_shard(data_path, sidecar_path, fingerprint)
        )
        shard_paths.append(data_path)
        job = {
            "index": index,
            "source": source,
            "config": config,
            "sensitivity_profiles": sensitivity_profiles,
            "map_filter": args.map_filter,
            "seed_filter": args.seed,
            "work_dir": work_dir,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
        }
        if reused:
            completed += 1
            print(
                f"[COLREGS-SOURCE] completed={completed}/{len(sources)} "
                f"method={source['method']} obs={source['count']} reused=True",
                flush=True,
            )
        else:
            pending.append(job)

    if int(args.workers) == 1:
        for job in pending:
            _write_source_shard(
                **job,
            )
            completed += 1
            source = job["source"]
            print(
                f"[COLREGS-SOURCE] completed={completed}/{len(sources)} "
                f"method={source['method']} obs={source['count']} reused=False",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(_write_source_shard, **job): job
                for job in pending
            }
            for future in as_completed(futures):
                future.result()
                completed += 1
                source = futures[future]["source"]
                print(
                    f"[COLREGS-SOURCE] completed={completed}/{len(sources)} "
                    f"method={source['method']} obs={source['count']} reused=False",
                    flush=True,
                )

    for data_path in shard_paths:
        result = _load_source_shard(data_path)
        all_events.extend(result.events)
        all_sensitivity_events.extend(result.sensitivity_events)
        episode_summaries.extend(result.episode_summaries)
        opportunity_digests.update(result.opportunity_digests)
        input_files.extend(result.input_files)
        for path, digest in result.map_inputs.items():
            if path not in map_inputs:
                map_inputs[path] = digest

    summary_rows = _event_summary_rows(all_events, episode_summaries)
    summary_rows.extend(summarize_unmatched_opportunities(
        all_events, opportunity_digests, config
    ))
    sensitivity_rows = _sensitivity_summary_rows(
        all_sensitivity_events,
        sensitivity_profiles,
        sorted({
            (str(row["method"]), int(row["obstacles"]))
            for row in episode_summaries
        }),
    )
    evaluator_path = Path(__file__).resolve()
    evaluator_root = evaluator_path.parents[1]
    manifest = {
        "trajectory_source": "ground_truth_dyn_traj",
        "methods": list(args.methods),
        "counts": list(args.counts),
        "episode_count": len(episode_summaries),
        "event_count": len(all_events),
        "censored_event_count": sum(
            bool(row.get("censored", False)) for row in all_events
        ),
        "target_boundary_hold_samples": sum(
            int(row["target_boundary_hold_samples"])
            for row in episode_summaries
        ),
        "summary_report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": _sha256(evaluator_path),
            "git_revision": _git_revision(evaluator_root),
        },
        "run_parameters": {
            "map_filter": args.map_filter,
            "seed_filter": args.seed,
        },
        "execution": {
            "workers": int(args.workers),
            "work_dir": str(work_dir),
            "resume_shards": not bool(args.no_resume_shards),
            "source_shards": len(shard_paths),
        },
        "event_field_units": {
            "t_start": "s",
            "t_action": "s",
            "t_cpa": "s",
            "t_end": "s",
            "initial_bearing": "deg_starboard_positive",
            "own_course": "deg_counterclockwise_positive",
            "target_course": "deg_counterclockwise_positive",
            "tcpa_start": "s",
            "dcpa_start": "m",
            "min_distance": "m",
            "course_change": "deg_rule_specific",
            "speed_reduction": "fraction",
            "response_ratio": "fraction",
        },
        "maps": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(map_inputs.items(), key=lambda item: str(item[0]))
        ],
        "configuration": {
            item.name: getattr(config, item.name) for item in fields(config)
        },
        "sensitivity_profiles": {
            name: {
                item.name: getattr(profile_config, item.name)
                for item in fields(profile_config)
            }
            for name, profile_config in sensitivity_profiles
        },
        "inputs": input_files,
    }
    _write_bundle(
        output_dir, all_events, episode_summaries,
        summary_rows, sensitivity_rows, manifest,
    )
    print(f"[DONE] COLREGs events: {len(all_events)}")
    print(f"[DONE] Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
