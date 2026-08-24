"""Shared evaluation helpers for final CW-VL/PPO experiment scripts."""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any

import numpy as np


def load_evaluation_protocol(
    path: str | pathlib.Path,
    *,
    repo_root: str | pathlib.Path,
) -> dict[str, Any]:
    """Load the shared comparison map/cache protocol."""
    root = pathlib.Path(repo_root).resolve()
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    episodes = int(data.get("episodes_per_density", 0))
    if episodes <= 0:
        raise ValueError("episodes_per_density must be positive")
    sources = []
    labels: set[str] = set()
    for item in data.get("sources", []):
        label = str(item["label"])
        if label in labels:
            raise ValueError(f"duplicate protocol label: {label}")
        labels.add(label)
        source = {
            "label": label,
            "obstacles": int(item["obstacles"]),
            "scenario_dir": pathlib.Path(item["scenario_dir"]),
            "kf_cache_dir": pathlib.Path(item["kf_cache_dir"]),
            "max_episode_steps": int(item["max_episode_steps"]),
        }
        for key in ("scenario_dir", "kf_cache_dir"):
            if not source[key].is_absolute():
                source[key] = root / source[key]
        if source["obstacles"] <= 0 or source["max_episode_steps"] <= 0:
            raise ValueError(f"invalid protocol source: {label}")
        sources.append(source)
    if not sources:
        raise ValueError("evaluation protocol has no sources")
    return {"episodes_per_density": episodes, "sources": sources}


def baseline_step_row(
    *,
    map_name: str,
    seed: int,
    step: int,
    env,
    info: dict[str, Any],
    fallback_action,
    latency: dict[str, Any],
) -> dict[str, Any]:
    """Build the lean step record shared by external baselines."""
    ship = np.asarray(env.ship_state, dtype=float)
    action = np.asarray(info.get("action", fallback_action), dtype=float).reshape(-1)
    row = {
        "map": str(map_name),
        "seed": int(seed),
        "step": int(step),
        "ship_x": float(info.get("ship_x", ship[0])),
        "ship_y": float(info.get("ship_y", ship[1])),
        "ship_psi": float(info.get("ship_psi", ship[2])),
        "ship_u": float(info.get("ship_u", ship[3])),
        "ship_v": float(info.get("ship_v", ship[4])),
        "ship_r": float(info.get("ship_r", ship[5])),
        "action_surge": float(action[0]) if action.size else float("nan"),
        "action_yaw": float(action[1]) if action.size > 1 else float("nan"),
    }
    row.update(latency)
    return row


def classify_episode_outcome(
    reason: Any,
    min_actual_distance: float,
    safety_distance: float = 2.0,
) -> dict[str, int]:
    """Return explicit 0/1 outcome flags for one episode."""
    reason_l = str(reason).lower()
    is_collision = any(
        kw in reason_l
        for kw in ("dynamic_obs", "static_obs", "out_of_bounds", "collision")
    )
    if is_collision:
        return {
            "raw_success": 0,
            "strict_success": 0,
            "collision": 1,
            "timeout": 0,
            "unsafe_near_miss": 0,
        }
    if reason == "goal_reached":
        strict_success = int(min_actual_distance >= safety_distance)
        return {
            "raw_success": 1,
            "strict_success": strict_success,
            "collision": 0,
            "timeout": 0,
            "unsafe_near_miss": int(not strict_success),
        }
    return {
        "raw_success": 0,
        "strict_success": 0,
        "collision": 0,
        "timeout": 1,
        "unsafe_near_miss": 0,
    }


def load_vn(path: str | pathlib.Path | None, *, missing_ok: bool = True):
    """Load a VecNormalize object enough to normalize dict observations."""
    if path is None or str(path).lower() == "none":
        return None
    vn_path = pathlib.Path(path)
    if not vn_path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"VecNormalize file not found: {vn_path}")
    with open(vn_path, "rb") as f:
        data = pickle.load(f)
    obs_rms = data.obs_rms if hasattr(data, "obs_rms") else data.get("obs_rms")

    class VN:
        def __init__(self, rms):
            self.rms = rms

        def normalize_obs(self, obs):
            normalized = {}
            for key, value in obs.items():
                if key == "dyn_mask":
                    normalized[key] = value
                elif key in self.rms:
                    normalized[key] = np.clip(
                        (value - self.rms[key].mean) / np.sqrt(self.rms[key].var + 1e-8),
                        -10,
                        10,
                    )
                else:
                    normalized[key] = value
            return normalized

    return VN(obs_rms)


def select_single_scenario_file(env, scenario_file: str | pathlib.Path) -> None:
    """Force the next reset of a freshly created env to load one scenario file."""
    env._scenario_files = [pathlib.Path(scenario_file)]
    env._sc_idx = 0
    env._shuffled_indices = None
    env._seq_pointer = 0


def load_scenario_into_env(env, scenario_file: str | pathlib.Path) -> None:
    """Load a single .npz scenario into an env configured with load_on_reset=False."""
    scenario_path = pathlib.Path(scenario_file)
    with np.load(scenario_path, allow_pickle=True) as data:
        required = ["grid", "init_pos", "init_psi", "goal", "dyn_traj"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{scenario_path} is missing required keys: {missing}")

        env.set_grid(data["grid"])
        env._current_scenario_path = scenario_path
        env.fixed_initial_position = data["init_pos"].astype(np.float32)
        env.fixed_initial_psi = float(data["init_psi"])
        env.fixed_goal = data["goal"].astype(np.float32)
        env.set_dyn_traj(data["dyn_traj"].astype(np.float32))
        env.dyn_seeds = data["dyn_seeds"] if "dyn_seeds" in data.files else None
