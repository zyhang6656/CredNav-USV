"""Algorithm-neutral scenario, checkpoint, and RNG helpers for trainers."""

from __future__ import annotations

import pathlib
import random
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv


def resolve_sources(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in config["data"][key]:
        scenario_dir = pathlib.Path(item["scenario_dir"])
        files = sorted(scenario_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"missing scenarios: {scenario_dir}")
        sources.append(
            {
                "label": str(item["label"]),
                "scenario_dir": scenario_dir,
                "kf_cache_dir": str(item["kf_cache_dir"]),
                "file_count": len(files),
            }
        )
    return sources


def _cleanup_partial_subproc_vec_env(env: SubprocVecEnv) -> None:
    remotes = list(getattr(env, "remotes", ()))
    work_remotes = list(getattr(env, "work_remotes", ()))
    processes = list(getattr(env, "processes", ()))
    for remote in remotes[: len(processes)]:
        try:
            remote.send(("close", None))
        except Exception:
            pass
    for remote in remotes + work_remotes:
        try:
            remote.close()
        except Exception:
            pass
    for process in processes:
        try:
            process.join(timeout=1.0)
        except Exception:
            pass
        try:
            alive = process.is_alive()
        except Exception:
            alive = True
        if alive:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.join(timeout=1.0)
            except Exception:
                pass
    env.closed = True


def _close_vec_env_quietly(env: Any) -> None:
    if isinstance(env, SubprocVecEnv):
        _cleanup_partial_subproc_vec_env(env)
        return
    try:
        env.close()
    except Exception:
        pass


def _torch_load(path: pathlib.Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _rng_state_dict() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _atomic_torch_save(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
