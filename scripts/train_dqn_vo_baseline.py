from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_obs3 import SaveCheckpointCallback, build_env_kwargs
from simple_boat.envs.dqn_vo_baseline import DQNVOObservationRewardWrapper
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def algorithm_signature(config: dict[str, Any]) -> dict[str, Any]:
    dqn = config["dqn"]
    action = config["dqn_vo"]
    return {
        "fixed_surge": float(action["fixed_surge"]),
        "yaw_command": float(action["yaw_command"]),
        "learning_rate": float(dqn["learning_rate"]),
        "gamma": float(dqn["gamma"]),
        "buffer_size": int(dqn["buffer_size"]),
        "batch_size": int(dqn["batch_size"]),
        "target_update_interval": int(dqn["target_update_interval"]),
        "exploration_final_eps": float(dqn["exploration_final_eps"]),
        "learning_starts": int(dqn["learning_starts"]),
        "policy_net_arch": [256, 256, 256, 256],
    }


def resolve_requested_steps(config: dict[str, Any], *, smoke: bool, override: int | None) -> int:
    if override is not None:
        if int(override) <= 0:
            raise ValueError("--max-timesteps must be positive")
        return int(override)
    key = "smoke_timesteps" if smoke else "max_timesteps"
    return int(config["dqn"][key])


def validate_resume_state(
    state: dict[str, Any],
    *,
    seed: int,
    max_steps: int,
    signature: dict[str, Any],
) -> None:
    if int(state.get("schema_version", 0)) != 1:
        raise ValueError("unsupported DRL-VO resume-state schema")
    if int(state["seed"]) != int(seed):
        raise ValueError(f"resume seed={state['seed']} does not match requested seed={seed}")
    if state["algorithm_signature"] != signature:
        raise ValueError("resume DRL-VO hyperparameters do not match current config")
    if int(state["steps"]) < 0 or int(state["steps"]) > int(max_steps):
        raise ValueError(f"resume step {state['steps']} is incompatible with max_steps={max_steps}")


def _atomic_replace(source: pathlib.Path, target: pathlib.Path) -> None:
    source.replace(target)


def save_resume_bundle(
    model: Any,
    *,
    run_dir: str | pathlib.Path,
    seed: int,
    signature: dict[str, Any],
    source_signature: tuple[tuple[Any, ...], ...],
) -> pathlib.Path:
    output = pathlib.Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "resume_model_latest.zip"
    replay_path = output / "resume_replay_latest.pkl"
    vec_path = output / "resume_vecnormalize_latest.pkl"
    state_path = output / "trainer_state_latest.pkl"
    model_tmp = output / "resume_model_latest.tmp.zip"
    replay_tmp = output / "resume_replay_latest.tmp.pkl"
    vec_tmp = output / "resume_vecnormalize_latest.tmp.pkl"
    state_tmp = output / "trainer_state_latest.tmp.pkl"

    model.save(model_tmp)
    model.save_replay_buffer(replay_tmp)
    vecnormalize = model.get_vec_normalize_env()
    if vecnormalize is None:
        raise RuntimeError("DRL-VO resumable training requires VecNormalize")
    vecnormalize.save(str(vec_tmp))
    env = model.get_env()
    if env is None:
        raise RuntimeError("DRL-VO resumable training requires an attached environment")
    env_states = env.env_method("get_resume_state")
    state = {
        "schema_version": 1,
        "seed": int(seed),
        "steps": int(model.num_timesteps),
        "algorithm_signature": signature,
        "source_signature": source_signature,
        "model": str(model_path.resolve()),
        "replay_buffer": str(replay_path.resolve()),
        "vecnormalize": str(vec_path.resolve()),
        "env_states": env_states,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    with state_tmp.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for source, target in (
        (model_tmp, model_path),
        (replay_tmp, replay_path),
        (vec_tmp, vec_path),
        (state_tmp, state_path),
    ):
        _atomic_replace(source, target)
    return state_path


def load_resume_state(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise ValueError("DRL-VO resume state must be a dictionary")
    return state


def source_signature(sources: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(source["label"]),
            str(pathlib.Path(source["scenario_dir"]).resolve()),
            str(pathlib.Path(source["kf_cache_dir"]).resolve()),
            int(source["file_count"]),
        )
        for source in sources
    )


def restore_rng_state(state: dict[str, Any]) -> None:
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


class SaveResumeBundleCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: pathlib.Path,
        save_freq: int,
        seed: int,
        signature: dict[str, Any],
        source_signature_value: tuple[tuple[Any, ...], ...],
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.save_freq = max(1, int(save_freq))
        self.seed = int(seed)
        self.signature = signature
        self.source_signature_value = source_signature_value
        self._pending = False

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._pending = True
        return True

    def _on_rollout_end(self) -> None:
        if self._pending:
            state_path = save_resume_bundle(
                self.model,
                run_dir=self.run_dir,
                seed=self.seed,
                signature=self.signature,
                source_signature=self.source_signature_value,
            )
            print(f"[TRAINER_STATE] step={self.num_timesteps} path={state_path}", flush=True)
            self._pending = False


def resolve_sources(config: dict, key: str) -> list[dict]:
    sources: list[dict] = []
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
            },
        )
    return sources


def resolve_train_sources(config: dict) -> list[dict]:
    return resolve_sources(config, "train_sources")


def env_config_for_dqn_vo(
    config: dict,
    kf_cache_dir: str,
    *,
    online_exact: bool = False,
) -> dict:
    cfg = dict(config)
    env_cfg = dict(cfg.get("env", {}))
    env_cfg["use_filter"] = True
    if online_exact:
        env_cfg["filter_execution_mode"] = "online_exact"
    cfg["env"] = env_cfg
    cfg["cache"] = {"kf_cache_dir": str(kf_cache_dir), "kf_cache_mode": "read_strict"}
    cfg["civo"] = {"enabled": False, "shield_enabled": False}
    cfg["rc_colregs"] = {"enabled": False, "reward_weight": 0.0}
    return cfg


def make_env(config: dict, source: dict, seed: int):
    def _init():
        kwargs = build_env_kwargs(env_config_for_dqn_vo(config, source["kf_cache_dir"]), source["scenario_dir"])
        kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
        env = USVEnvMinimal(**kwargs)
        env = DQNVOObservationRewardWrapper(
            env,
            fixed_surge=float(config["dqn_vo"]["fixed_surge"]),
            yaw_command=float(config["dqn_vo"]["yaw_command"]),
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dqn_vo_baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--resume", type=pathlib.Path, default=None)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resume_state = load_resume_state(args.resume) if args.resume is not None else None
    run_dir = (
        args.resume.resolve().parent
        if args.resume is not None
        else pathlib.Path(config["run"]["output_root"]) / f"{config['run']['name']}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_train_sources(config)
    n_envs = len(sources)
    steps = resolve_requested_steps(config, smoke=bool(args.smoke), override=args.max_timesteps)
    signature = algorithm_signature(config)
    source_signature_value = source_signature(sources)
    if resume_state is not None:
        validate_resume_state(resume_state, seed=seed, max_steps=steps, signature=signature)
        if tuple(tuple(item) for item in resume_state["source_signature"]) != source_signature_value:
            raise ValueError("resume DRL-VO training sources do not match current sources")

    raw_env = DummyVecEnv([make_env(config, source, seed + i) for i, source in enumerate(sources)])
    if resume_state is None:
        env = VecNormalize(
            raw_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=float(config["dqn"]["gamma"]),
        )
    else:
        env = VecNormalize.load(resume_state["vecnormalize"], raw_env)
        env.training = True
        env.norm_reward = True
    checkpoint_freq_steps = int(config["dqn"]["checkpoint_freq_steps"])
    checkpoint_callback = SaveCheckpointCallback(
        str(run_dir / "checkpoints"),
        max(1, checkpoint_freq_steps // max(1, n_envs)),
        verbose=0,
    )
    resume_callback = SaveResumeBundleCallback(
        run_dir=run_dir,
        save_freq=max(1, checkpoint_freq_steps // max(1, n_envs)),
        seed=seed,
        signature=signature,
        source_signature_value=source_signature_value,
    )
    if resume_state is None:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=float(config["dqn"]["learning_rate"]),
            gamma=float(config["dqn"]["gamma"]),
            buffer_size=int(config["dqn"]["buffer_size"]),
            batch_size=int(config["dqn"]["batch_size"]),
            target_update_interval=int(config["dqn"]["target_update_interval"]),
            exploration_final_eps=float(config["dqn"]["exploration_final_eps"]),
            learning_starts=int(config["dqn"]["learning_starts"]),
            policy_kwargs={"net_arch": [256, 256, 256, 256]},
            verbose=1,
            seed=seed,
        )
        initial_step = 0
    else:
        model = DQN.load(resume_state["model"], env=env, force_reset=False)
        model.load_replay_buffer(resume_state["replay_buffer"])
        for index, env_state in enumerate(resume_state["env_states"]):
            env.env_method("set_resume_state", env_state, indices=index)
        restore_rng_state(resume_state)
        initial_step = int(resume_state["steps"])
        if int(model.num_timesteps) != initial_step:
            raise ValueError(
                f"resume model timestep={model.num_timesteps} does not match state step={initial_step}"
            )
        print(f"[RESUME_FULL] {args.resume} at step {initial_step}", flush=True)
    remaining_steps = steps - initial_step
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=CallbackList([checkpoint_callback, resume_callback]),
            reset_num_timesteps=resume_state is None,
        )

    model_path = run_dir / f"model_{steps}.zip"
    vec_path = run_dir / f"vecnormalize_{steps}.pkl"
    model.save(model_path)
    env.save(str(vec_path))
    state_path = save_resume_bundle(
        model,
        run_dir=run_dir,
        seed=seed,
        signature=signature,
        source_signature=source_signature_value,
    )
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "steps": steps,
                "config": str(config_path),
                "mix_mode": str(config["data"].get("train_mix_mode", "parallel_equal_obs_count_sources")),
                "n_envs": n_envs,
                "train_sources": [
                    {
                        "label": source["label"],
                        "scenario_dir": str(source["scenario_dir"]),
                        "kf_cache_dir": source["kf_cache_dir"],
                        "file_count": source["file_count"],
                    }
                    for source in sources
                ],
                "checkpoint_freq_steps": checkpoint_freq_steps,
                "model": str(model_path),
                "vecnormalize": str(vec_path),
                "trainer_state": str(state_path),
                "resumed_from": str(args.resume) if args.resume is not None else None,
                "algorithm_signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
