from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import signal
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training_io import (
    _atomic_torch_save,
    _close_vec_env_quietly,
    _restore_rng_state,
    _rng_state_dict,
    _torch_load,
    resolve_sources,
)
from scripts.train_obs3 import build_env_kwargs
from simple_boat.envs.lag_u_baseline import (
    LagUAgent,
    LagUReplayBuffer,
    TrustFreeObservationWrapper,
    transition_cost,
)
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def env_config_for_lag_u(
    config: dict[str, Any],
    kf_cache_dir: str,
    *,
    online_exact: bool = False,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["env"] = dict(resolved.get("env", {}), use_filter=True)
    if online_exact:
        resolved["env"]["filter_execution_mode"] = "online_exact"
    resolved["cache"] = {"kf_cache_dir": str(kf_cache_dir), "kf_cache_mode": "read_strict"}
    resolved["civo"] = {"enabled": False, "shield_enabled": False}
    resolved["rc_colregs"] = {"enabled": False, "reward_weight": 0.0}
    return resolved


def validate_training_config(config: dict[str, Any], *, source_count: int) -> None:
    cfg = config["lag_u"]
    n_envs = int(cfg.get("n_envs", source_count))
    max_steps = int(cfg.get("max_timesteps", 5_000_000))
    if n_envs != int(source_count):
        raise ValueError(f"lag_u.n_envs={n_envs} must equal train source count {source_count}")
    if n_envs <= 0 or max_steps <= 0:
        raise ValueError("n_envs and max_timesteps must be positive")
    if max_steps % n_envs != 0:
        raise ValueError(f"max_timesteps={max_steps} must be divisible by n_envs={n_envs}")
    if int(cfg.get("train_freq_steps", 1)) != 1:
        raise ValueError("Lag-U reproduction requires train_freq_steps=1")
    if int(cfg.get("ensemble_members", 3)) != 3:
        raise ValueError("paper Lag-U requires ensemble_members=3")
    if int(cfg.get("learning_starts", 25_000)) < int(cfg.get("batch_size", 256)):
        raise ValueError("learning_starts must be at least batch_size")
    if float(cfg.get("gamma", 0.99)) != 0.99:
        raise ValueError("paper Lag-U requires gamma=0.99")
    if float(cfg.get("cost_gamma", 0.99)) != 0.99:
        raise ValueError("paper Lag-U requires cost_gamma=0.99")
    if cfg.get("curriculum", {}).get("enabled", False):
        raise ValueError("Lag-U baseline must not use curriculum learning")


def _make_env(config: dict[str, Any], source: dict[str, Any], seed: int) -> TrustFreeObservationWrapper:
    kwargs = build_env_kwargs(env_config_for_lag_u(config, source["kf_cache_dir"]), source["scenario_dir"])
    kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    env = TrustFreeObservationWrapper(USVEnvMinimal(**kwargs))
    env.reset(seed=int(seed))
    return env


def _make_env_fn(
    config: dict[str, Any],
    source: dict[str, Any],
    seed: int,
) -> Callable[[], TrustFreeObservationWrapper]:
    def _init() -> TrustFreeObservationWrapper:
        return _make_env(config, source, seed)

    return _init


def make_train_vec_env(config: dict[str, Any], sources: list[dict[str, Any]], seed: int):
    env_fns = [_make_env_fn(config, source, seed + index) for index, source in enumerate(sources)]
    if len(env_fns) > 1:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def _source_signature(sources: list[dict[str, Any]]) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        (
            str(source["label"]),
            pathlib.Path(source["scenario_dir"]).as_posix(),
            pathlib.Path(source["kf_cache_dir"]).as_posix(),
            int(source["file_count"]),
        )
        for source in sources
    )


def _algorithm_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "hidden_dims",
        "ensemble_members",
        "actor_learning_rate",
        "critic_learning_rate",
        "lagrange_learning_rate",
        "gamma",
        "cost_gamma",
        "base_cost_bound",
        "uncertainty_threshold",
        "exploration_tc",
        "exploration_minimum",
        "batch_size",
        "buffer_size",
        "learning_starts",
        "train_freq_steps",
        "policy_delay",
        "target_tau",
        "target_policy_noise",
        "target_noise_clip",
        "n_envs",
    )
    return {key: copy.deepcopy(cfg.get(key)) for key in keys}


def _build_agent(cfg: dict[str, Any], obs_dim: int, action_dim: int, device: torch.device) -> LagUAgent:
    return LagUAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(int(value) for value in cfg.get("hidden_dims", [256, 256])),
        ensemble_members=int(cfg.get("ensemble_members", 3)),
        actor_lr=float(cfg.get("actor_learning_rate", 3e-4)),
        critic_lr=float(cfg.get("critic_learning_rate", 3e-4)),
        lagrange_lr=float(cfg.get("lagrange_learning_rate", 1e-5)),
        gamma=float(cfg.get("gamma", 0.99)),
        cost_gamma=float(cfg.get("cost_gamma", 0.99)),
        tau=float(cfg.get("target_tau", 0.005)),
        policy_delay=int(cfg.get("policy_delay", 2)),
        target_policy_noise=float(cfg.get("target_policy_noise", 0.2)),
        target_noise_clip=float(cfg.get("target_noise_clip", 0.5)),
        base_cost_bound=float(cfg.get("base_cost_bound", 0.1)),
        uncertainty_threshold=float(cfg.get("uncertainty_threshold", 0.07)),
        exploration_tc=int(cfg.get("exploration_tc", 300_000)),
        exploration_minimum=float(cfg.get("exploration_minimum", 0.1)),
        device=device,
    )


def _save_model(
    path: pathlib.Path,
    agent: LagUAgent,
    *,
    config: dict[str, Any],
    seed: int,
    steps: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    _atomic_torch_save(
        {
            "model_version": 2,
            "actor": agent.actor.state_dict(),
            "reward_critic": agent.reward_critic.state_dict(),
            "cost_critic": agent.cost_critic.state_dict(),
            "config": config,
            "seed": int(seed),
            "steps": int(steps),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
        },
        path,
    )


def train(
    config: dict[str, Any],
    *,
    seed: int,
    smoke: bool = False,
    resume: pathlib.Path | None = None,
) -> pathlib.Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = config["lag_u"]
    sources = resolve_sources(config, "train_sources")
    validate_training_config(config, source_count=len(sources))
    n_envs = int(cfg["n_envs"])
    max_steps = int(cfg.get("smoke_timesteps", 128) if smoke else cfg["max_timesteps"])
    if max_steps % n_envs != 0:
        raise ValueError(f"requested max steps {max_steps} must be divisible by n_envs={n_envs}")

    requested_device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    checkpoint_data = _torch_load(resume, device) if resume is not None else None
    initial_step = int(checkpoint_data["steps"]) if checkpoint_data is not None else 0
    if initial_step < 0 or initial_step > max_steps or (max_steps - initial_step) % n_envs != 0:
        raise ValueError(f"checkpoint step {initial_step} is incompatible with max_steps={max_steps}")

    run_dir = (
        pathlib.Path(resume).resolve().parent
        if resume is not None
        else pathlib.Path(config["run"]["output_root"])
        / f"{config['run']['name']}_{'smoke_' if smoke else ''}seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_train_vec_env(config, sources, seed)
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[STOP] signal={signum}; finishing current VecEnv step before saving", flush=True)

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, request_stop)
        except (OSError, ValueError):
            pass

    current_step = initial_step
    state_is_aligned = True
    try:
        obs_list = np.asarray(env.reset(), dtype=np.float32)
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        agent = _build_agent(cfg, obs_dim, action_dim, device)
        replay = LagUReplayBuffer(
            capacity=int(cfg.get("smoke_buffer_size", 1024) if smoke else cfg.get("buffer_size", 1_000_000)),
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=seed + 101,
        )
        batch_size = int(cfg.get("smoke_batch_size", 16) if smoke else cfg.get("batch_size", 256))
        learning_starts = int(
            cfg.get("smoke_learning_starts", 32) if smoke else cfg.get("learning_starts", 25_000)
        )
        if learning_starts < batch_size:
            raise ValueError("effective learning_starts must be at least effective batch_size")
        episode_returns = [0.0] * n_envs
        episode_costs = [0.0] * n_envs
        stats: list[dict[str, Any]] = []

        if checkpoint_data is not None:
            if int(checkpoint_data.get("trainer_state_version", 0)) != 2:
                raise ValueError("resume requires a linear-cost Lag-U full trainer state v2")
            if int(checkpoint_data["seed"]) != int(seed):
                raise ValueError(f"checkpoint seed={checkpoint_data['seed']} does not match seed={seed}")
            if tuple(tuple(item) for item in checkpoint_data["source_signature"]) != _source_signature(sources):
                raise ValueError("checkpoint training sources do not match current sources")
            if checkpoint_data["algorithm_signature"] != _algorithm_signature(cfg):
                raise ValueError("checkpoint Lag-U hyperparameters do not match current config")
            if int(checkpoint_data["obs_dim"]) != obs_dim or int(checkpoint_data["action_dim"]) != action_dim:
                raise ValueError("checkpoint observation/action dimensions do not match the environment")
            agent.load_state_dict(checkpoint_data["agent"])
            replay.load_state_dict(checkpoint_data["replay"])
            obs_list = np.asarray(checkpoint_data["obs_list"], dtype=np.float32)
            episode_returns = [float(value) for value in checkpoint_data["episode_returns"]]
            episode_costs = [float(value) for value in checkpoint_data["episode_costs"]]
            stats = list(checkpoint_data["stats"])
            for index, env_state in enumerate(checkpoint_data["env_states"]):
                env.env_method("set_resume_state", env_state, indices=index)
            _restore_rng_state(checkpoint_data["rng"])
            print(f"[RESUME_FULL] {resume} at step {initial_step}", flush=True)

        checkpoint_freq = int(cfg.get("checkpoint_freq_steps", 500_000))
        state_freq = int(cfg.get("state_checkpoint_freq_steps", checkpoint_freq))
        log_freq = int(cfg.get("log_freq_steps", 10_000))
        started_at = time.perf_counter()
        latest_diagnostics: dict[str, float | bool] = {}
        last_state_step = -1
        last_model_step = -1

        def save_full_state(step_value: int) -> None:
            nonlocal last_state_step
            env_states = env.env_method("get_resume_state")
            _atomic_torch_save(
                {
                    "trainer_state_version": 2,
                    "seed": int(seed),
                    "steps": int(step_value),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "source_signature": _source_signature(sources),
                    "algorithm_signature": _algorithm_signature(cfg),
                    "agent": agent.state_dict(),
                    "replay": replay.state_dict(),
                    "obs_list": np.asarray(obs_list, dtype=np.float32),
                    "episode_returns": list(episode_returns),
                    "episode_costs": list(episode_costs),
                    "stats": list(stats),
                    "env_states": env_states,
                    "rng": _rng_state_dict(),
                    "config": config,
                },
                run_dir / "trainer_state_latest.pt",
            )
            last_state_step = int(step_value)

        for vector_step in range(initial_step, max_steps, n_envs):
            if vector_step < learning_starts:
                actions = np.random.uniform(-1.0, 1.0, size=(n_envs, action_dim)).astype(np.float32)
            else:
                actions = np.asarray(agent.act(obs_list), dtype=np.float32)

            state_is_aligned = False
            next_obs_batch, rewards, dones, infos = env.step(actions)
            next_obs_batch = np.asarray(next_obs_batch, dtype=np.float32)
            for env_index in range(n_envs):
                transition_step = vector_step + env_index + 1
                info = infos[env_index]
                done = bool(dones[env_index])
                terminal_obs = info.get("terminal_observation")
                replay_next_obs = np.asarray(
                    next_obs_batch[env_index] if terminal_obs is None else terminal_obs,
                    dtype=np.float32,
                )
                cost = transition_cost(info)
                replay.add_batch(
                    obs=obs_list[env_index : env_index + 1],
                    actions=actions[env_index : env_index + 1],
                    rewards=np.asarray([rewards[env_index]], dtype=np.float32),
                    costs=np.asarray([cost], dtype=np.float32),
                    next_obs=replay_next_obs.reshape(1, -1),
                    dones=np.asarray([float(done)], dtype=np.float32),
                )
                episode_returns[env_index] += float(rewards[env_index])
                episode_costs[env_index] += cost
                if replay.size >= learning_starts:
                    latest_diagnostics.update(
                        agent.update(
                            replay.sample(batch_size, device=device),
                            environment_step=transition_step,
                        )
                    )
                if done:
                    stats.append(
                        {
                            "step": transition_step,
                            "env": env_index,
                            "source": str(sources[env_index]["label"]),
                            "return": float(episode_returns[env_index]),
                            "cost": float(episode_costs[env_index]),
                            "reason": str(info.get("reason", info.get("timeout_reason", ""))),
                        }
                    )
                    episode_returns[env_index] = 0.0
                    episode_costs[env_index] = 0.0

            obs_list = next_obs_batch
            current_step = vector_step + n_envs
            state_is_aligned = True
            if log_freq > 0 and current_step % log_freq == 0:
                recent = stats[-20:]
                recent_return = float(np.mean([row["return"] for row in recent])) if recent else float("nan")
                recent_cost = float(np.mean([row["cost"] for row in recent])) if recent else float("nan")
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                fps = (current_step - initial_step) / elapsed
                print(
                    f"[TRAIN] step={current_step}/{max_steps} buffer={replay.size} episodes={len(stats)} "
                    f"recent_return={recent_return:.2f} recent_cost={recent_cost:.3f} "
                    f"lambda={agent.lagrange.item():.5f} fps={fps:.1f} "
                    f"q_nonpositive={latest_diagnostics.get('nonpositive_q_mean_rate', float('nan')):.3f} "
                    f"q_std={latest_diagnostics.get('q_std_mean', float('nan')):.3f} "
                    f"ratio_p95={latest_diagnostics.get('uncertainty_ratio_p95', float('nan')):.3f} "
                    f"cost_q={latest_diagnostics.get('cost_q_mean', float('nan')):.3f} "
                    f"bound={latest_diagnostics.get('cost_bound_mean', float('nan')):.3f}",
                    flush=True,
                )
            if state_freq > 0 and current_step % state_freq == 0:
                save_full_state(current_step)
                print(f"[TRAINER_STATE] step={current_step}", flush=True)
            if checkpoint_freq > 0 and current_step % checkpoint_freq == 0:
                model_path = run_dir / f"lag_u_{current_step}.pt"
                _save_model(
                    model_path,
                    agent,
                    config=config,
                    seed=seed,
                    steps=current_step,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )
                last_model_step = current_step
                (run_dir / "episodes.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                print(f"[CHECKPOINT] {model_path}", flush=True)
            if stop_requested:
                break

        model_path = run_dir / f"lag_u_{current_step}.pt"
        if last_model_step != current_step:
            _save_model(
                model_path,
                agent,
                config=config,
                seed=seed,
                steps=current_step,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
        if last_state_step != current_step:
            save_full_state(current_step)
        (run_dir / "episodes.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "algorithm": "Lag-U-USV",
                    "paper": "Safe Reinforcement Learning in Autonomous Driving With Epistemic Uncertainty Estimation",
                    "doi": "10.1109/TITS.2024.3397700",
                    "primary_evaluation": "pure Lag-U-USV without intervention",
                    "observation_dim": obs_dim,
                    "trust_in_policy_observation": False,
                    "curriculum": False,
                    "civo": False,
                    "rc_colregs": False,
                    "reward": "shared USV environment reward",
                    "adaptations": {
                        "signed_reward_uncertainty": "sqrt(Q_eu) / (abs(Q_mean) + epsilon)",
                        "domain_cost_mapping": "terminal dynamic/static/out-of-bounds event maps to cost 1",
                    },
                    "cost": "terminal event cost 1 for dynamic/static/out-of-bounds collision, else 0",
                    "cost_discount": "paper gamma_c=0.99",
                    "cost_output": "paper linear discounted cumulative Cost-Q",
                    "ensemble_current_output": "primary critic Q1 from each independently initialized twin-critic member",
                    "reward_target": "minimum of each member's two target critics with clipped target-policy noise",
                    "cost_target": "single target cost critic with the clean target-policy action",
                    "paper_omitted_td3_assumptions": {
                        "batch_size": int(cfg.get("batch_size", 256)),
                        "buffer_size": int(cfg.get("buffer_size", 1_000_000)),
                        "random_warmup_steps": int(cfg.get("learning_starts", 25_000)),
                        "policy_delay": int(cfg.get("policy_delay", 2)),
                        "target_tau": float(cfg.get("target_tau", 0.005)),
                        "target_policy_noise": float(cfg.get("target_policy_noise", 0.2)),
                        "target_noise_clip": float(cfg.get("target_noise_clip", 0.5)),
                        "target_update": "target=(1-tau)*target+tau*online",
                        "initial_lagrange_multiplier": 0.0,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return model_path
    except BaseException:
        if current_step > initial_step and state_is_aligned:
            try:
                save_full_state(current_step)
                print(f"[EMERGENCY_STATE] step={current_step}", flush=True)
            except BaseException as save_error:
                print(f"[WARN] emergency state save failed: {save_error}", flush=True)
        raise
    finally:
        _close_vec_env_quietly(env)
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/lag_u_baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-timesteps", type=int, default=None)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["run"].get("seed", 0) if args.seed is None else args.seed)
    if args.max_timesteps is not None:
        target_key = "smoke_timesteps" if args.smoke else "max_timesteps"
        config["lag_u"][target_key] = int(args.max_timesteps)
    checkpoint = train(
        config,
        seed=seed,
        smoke=bool(args.smoke),
        resume=pathlib.Path(args.resume) if args.resume else None,
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
