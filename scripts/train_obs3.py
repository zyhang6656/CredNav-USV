"""Stage O3-train: PPO from scratch on obs3 with detailed metrics."""

import argparse
import json
import os
import pathlib
import sys
import yaml
import multiprocessing
from typing import Callable

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.utils import set_random_seed

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def linear_floor_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * max(0.0, progress_remaining)
    return func


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    return linear_floor_schedule(initial_value, 0.0)


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    return linear_floor_schedule(initial_value, initial_value)


class SyncVecNormalizeCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        train_env = self.model.get_vec_normalize_env()
        if train_env is not None and self.eval_env is not None:
            sync_envs_normalization(train_env, self.eval_env)
        return True


class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = pathlib.Path(save_dir)
        self.save_freq = save_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            path = self.save_dir / f"checkpoint_{step:07d}"
            vn = self.model.get_vec_normalize_env()
            self.model.save(str(path))
            if vn:
                vn.save(str(self.save_dir / f"vecnormalize_{step:07d}.pkl"))
        return True


def make_env(rank: int, seed: int, env_class, env_kwargs: dict) -> Callable:
    def _init():
        set_random_seed(seed + rank)
        env = env_class(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_vec_env(n_envs: int, seed: int, env_class, env_kwargs: dict, use_subproc: bool = True):
    env_fns = [make_env(i, seed, env_class, env_kwargs) for i in range(n_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns, start_method="spawn")
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv(env_fns)


def make_vec_env_from_kwargs_list(seed: int, env_class, env_kwargs_list: list[dict], use_subproc: bool = True):
    env_fns = [make_env(i, seed, env_class, kwargs) for i, kwargs in enumerate(env_kwargs_list)]
    if use_subproc and len(env_fns) > 1:
        return SubprocVecEnv(env_fns, start_method="spawn")
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv(env_fns)


def resolve_scratch_dirs(config: dict) -> tuple[pathlib.Path, pathlib.Path]:
    data_cfg = config.get("data", {})
    return (
        pathlib.Path(data_cfg.get("train_scenario_dir", "./simple_boat/assets/nav3_new_map")),
        pathlib.Path(data_cfg.get("eval_scenario_dir", "./simple_boat/assets/eval3_new_map")),
    )


def resolve_train_sources(config: dict) -> list[dict]:
    sources = []
    for item in config.get("data", {}).get("train_sources", []) or []:
        scenario_dir = pathlib.Path(item["scenario_dir"])
        files = sorted(scenario_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"missing train source scenarios: {scenario_dir}")
        sources.append({
            "label": str(item["label"]),
            "scenario_dir": scenario_dir,
            "kf_cache_dir": str(item["kf_cache_dir"]),
            "kf_cache_mode": str(item.get("kf_cache_mode", "read_strict")),
            "file_count": len(files),
        })
    return sources


def resolve_scratch_run(config: dict, seed: int) -> tuple[pathlib.Path, str]:
    exp_name = config.get("experiment_name", "obs3_ppo_scratch")
    run_cfg = config.get("run", {})
    run_tag = str(run_cfg.get("run_tag", f"ppo_obs3_seed{seed}"))
    run_root = pathlib.Path(run_cfg.get("run_root", f"./runs/obs3/{exp_name}"))
    return run_root / run_tag, run_tag


def write_run_metadata(
    *,
    run_dir: pathlib.Path,
    config: dict,
    seed: int,
    config_source: pathlib.Path,
) -> None:
    """Persist the inputs needed to reproduce a training run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "entrypoint": "scripts/train_obs3.py",
        "config_source": config_source.as_posix(),
        "seed": int(seed),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def resolve_value_loss_type(config: dict) -> str:
    algo = str(config.get("algo", "ppo")).lower()
    modes = {"ppo": "mse", "hetero_ppo": "hetero", "cwvl": "cwvl"}
    if algo not in modes:
        raise ValueError(f"algo must be 'ppo', 'hetero_ppo', or 'cwvl', got {algo!r}")
    return modes[algo]


def build_env_kwargs(config: dict, scenario_dir: pathlib.Path) -> dict:
    env_cfg = config.get("env", {})
    noise_cfg = config.get("noise", {})
    trust_cfg = config.get("trust", {})
    reward_cfg = config.get("reward", {})
    cache_cfg = config.get("cache", {})
    civo_cfg = config.get("civo", {})
    colregs_cfg = config.get("rc_colregs", config.get("colregs", {}))
    if "vo_cbf_prediction_steps" in civo_cfg:
        raise ValueError("CBF-VO prediction horizon is fixed to one step")
    retired_corecbf_keys = sorted(
        {
            "eta",
            "qp_solver",
            "hocbf_gamma",
            "hocbf_margin_gate",
            "recovery_accel",
            "hocbf_qp_enabled",
            "qp_slack_weight",
            "qp_dot_tau_u_weight",
            "min_surge_action",
            "nominal_min_surge_action",
            "hocbf_gain",
            "barrier_distance",
        }.intersection(civo_cfg)
    )
    if retired_corecbf_keys:
        raise ValueError(
            f"retired CoReCBF config key(s): {', '.join(retired_corecbf_keys)}; "
            "use surge_accel, turn_accel, turn_direction, cbf_gain, and safety_distance"
        )

    kwargs = dict(
        scenario_dir=scenario_dir,
        load_on_reset=True,
        max_episode_steps=int(env_cfg.get("max_episode_steps", 512)),
        dt=float(env_cfg.get("dt", 0.1)),
        render_freq=5,
        render_mode=False,
        dynamic_obstacles=True,
        init_goal_threshold=float(env_cfg.get("init_goal_threshold", 1.0)),
        init_collision_threshold=float(env_cfg.get("init_collision_threshold", 1.6)),
        use_filter=bool(env_cfg.get("use_filter", True)),
        filter_execution_mode=str(
            env_cfg.get("filter_execution_mode", "precomputed")
        ),
        actuator_tau_u_dot_max=(
            None
            if env_cfg.get("actuator_tau_u_dot_max", None) is None
            else float(env_cfg.get("actuator_tau_u_dot_max"))
        ),
        actuator_n_dot_max=(
            None
            if env_cfg.get("actuator_n_dot_max", None) is None
            else float(env_cfg.get("actuator_n_dot_max"))
        ),
        # Noise: disabled
        burst_enabled=bool(noise_cfg.get("burst_enabled", False)),
        burst_episode_prob=float(noise_cfg.get("burst_episode_prob", 0.0)),
        burst_duration_steps=int(noise_cfg.get("burst_duration_steps", 0)),
        measurement_cov_scale=float(noise_cfg.get("measurement_cov_scale", 0)),
        bias_enabled=bool(noise_cfg.get("bias_enabled", False)),
        bias_position_y=float(noise_cfg.get("bias_position_y", 0.75)),
        bias_duration_steps=int(noise_cfg.get("bias_duration_steps", 30)),
        burst_start_mode=str(noise_cfg.get("start_mode", "risk_onset")),
        measurement_delay_steps=int(noise_cfg.get("measurement_delay_steps", 0)),
        nominal_position_std=float(noise_cfg.get("nominal_position_std", 0.10)),
        nominal_velocity_std=float(noise_cfg.get("nominal_velocity_std", 0.03)),
        path_progress_arrival_steps=int(noise_cfg.get("path_progress_arrival_steps", 230)),
        path_progress_candidate_stride=int(noise_cfg.get("path_progress_candidate_stride", 1)),
        path_progress_max_pulses=int(noise_cfg.get("path_progress_max_pulses", 1)),
        path_progress_topk=int(noise_cfg.get("path_progress_topk", 1)),
        path_progress_min_separation_steps=int(noise_cfg.get("path_progress_min_separation_steps", 60)),
        path_progress_anchor_lead_steps=int(noise_cfg.get("path_progress_anchor_lead_steps", 0)),
        path_progress_candidate_distance_threshold=(
            None
            if noise_cfg.get("path_progress_candidate_distance_threshold", None) is None
            else float(noise_cfg.get("path_progress_candidate_distance_threshold"))
        ),
        kf_cache_dir=cache_cfg.get("kf_cache_dir", None),
        kf_cache_mode=str(cache_cfg.get("kf_cache_mode", "off")),
        # Trust
        trust_mode=str(trust_cfg.get("mode", "oracle_tmse")),
        trust_window_size=int(trust_cfg.get("window_size", 20)),
        trust_t_min=float(trust_cfg.get("t_min", 0.05)),
        trust_t_max=float(trust_cfg.get("t_max", 1.0)),
        trust_aggregate=str(trust_cfg.get("aggregate", "min_over_risk_active")),
        trust_innovation_mode=str(trust_cfg.get("innovation_mode", "position_only_2d")),
        risk_distance_threshold=float(trust_cfg.get("risk_distance_threshold", 10.0)),
        tcpa_horizon=float(trust_cfg.get("tcpa_horizon", 12.0)),
        # Reward
        success_bonus=float(reward_cfg.get("success_bonus", 200.0)),
        collision_penalty=float(reward_cfg.get("collision_penalty", -300.0)),
        timeout_penalty=float(reward_cfg.get("timeout_penalty", -50.0)),
        progress_weight=float(reward_cfg.get("progress_weight", 30.0)),
        cte_penalty_weight=float(reward_cfg.get("cte_penalty_weight", 0.1)),
        time_penalty=float(reward_cfg.get("time_penalty", -0.01)),
        near_risk_weight=float(reward_cfg.get("near_risk_weight", 3.0)),
        actual_clearance_weight=float(reward_cfg.get("actual_clearance_weight", 0.0)),
        actual_clearance_margin=float(reward_cfg.get("actual_clearance_margin", 2.0)),
        # Optional eval-time CI-VO shield. Defaults preserve existing runs.
        civo_enabled=bool(civo_cfg.get("enabled", False) or civo_cfg.get("shield_enabled", False)),
        civo_confidence=float(civo_cfg.get("confidence", 0.99)),
        civo_shield_enabled=bool(civo_cfg.get("shield_enabled", False)),
        civo_shield_distance=float(civo_cfg.get("shield_distance", 10.0)),
        civo_shield_tcpa_horizon=float(civo_cfg.get("shield_tcpa_horizon", 12.0)),
        civo_shield_method=str(civo_cfg.get("shield_method", "corecbf")),
        civo_shield_gate_mode=str(
            civo_cfg.get("shield_gate_mode", "distance_tcpa")
        ),
        vo_cbf_alpha_vo=float(civo_cfg.get("vo_cbf_alpha_vo", 10.0)),
        vo_cbf_alpha_c=float(civo_cfg.get("vo_cbf_alpha_c", 10.0)),
        vo_cbf_k_u=float(civo_cfg.get("vo_cbf_k_u", 2.0)),
        vo_cbf_k_vo=float(civo_cfg.get("vo_cbf_k_vo", 50.0)),
        corecbf_variant=str(civo_cfg.get("corecbf_variant", "deterministic")),
        corecbf_surge_accel=float(civo_cfg.get("surge_accel", 1.0)),
        corecbf_turn_accel=float(civo_cfg.get("turn_accel", 1.0)),
        corecbf_turn_direction=int(civo_cfg.get("turn_direction", -1)),
        corecbf_gain=float(civo_cfg.get("cbf_gain", 1.0)),
        corecbf_safety_distance=float(civo_cfg.get("safety_distance", 2.0)),
        corecbf_colregs_reference_scale=float(
            civo_cfg.get("colregs_reference_scale", 0.1875)
        ),
        corecbf_tau_u_weight=float(civo_cfg.get("qp_tau_u_weight", 1.0)),
        corecbf_tau_r_weight=float(civo_cfg.get("qp_tau_r_weight", 1.0)),
        corecbf_separable_slack_enabled=bool(
            civo_cfg.get("separable_slack_enabled", False)
        ),
        corecbf_slack_weight=float(civo_cfg.get("shared_slack_weight", 1.0e4)),
        corecbf_osqp_max_iter=int(civo_cfg.get("osqp_max_iter", 4000)),
        corecbf_osqp_eps_abs=float(civo_cfg.get("osqp_eps_abs", 1e-5)),
        corecbf_osqp_eps_rel=float(civo_cfg.get("osqp_eps_rel", 1e-5)),
        corecbf_osqp_polishing=bool(civo_cfg.get("osqp_polishing", True)),
        # Optional RC-COLREGs metrics/reward. Defaults preserve existing runs.
        rc_colregs_enabled=bool(colregs_cfg.get("enabled", False)),
        rc_colregs_reward_weight=float(colregs_cfg.get("reward_weight", 0.0)),
        rc_colregs_d_safe=float(colregs_cfg.get("d_safe", 3.0)),
        rc_colregs_tau=float(colregs_cfg.get("tau", 10.0)),
        rc_colregs_kappa=float(colregs_cfg.get("kappa", 60.0)),
        rc_colregs_kappa_beta=float(colregs_cfg.get("kappa_beta", colregs_cfg.get("kappa", 60.0))),
        rc_colregs_kappa_time=float(colregs_cfg.get("kappa_time", colregs_cfg.get("kappa", 60.0))),
    )
    kwargs["trigger_on_risk_active"] = True
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/obs3/ppo_scratch.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ppo_cfg = config.get("ppo", {})
    critic_cfg = config.get("critic", {})
    features_cfg = config.get("features", {})
    eval_cfg = config.get("eval", {})

    from simple_boat.envs.usv_env_minimal import USVEnvMinimal
    from simple_boat.envs.custom_feature_extractor_minimal import MinimalFeatureExtractor
    from simple_boat.envs.custom_ppo_policy_unified import UnifiedPolicy
    from simple_boat.envs.unified_ppo_trainer import UnifiedPPOTrainer
    from simple_boat.envs.eval_callback import TrainingMetricsCallback

    train_sources = resolve_train_sources(config)
    train_scenario_dir, eval_scenario_dir = resolve_scratch_dirs(config)

    if train_sources:
        if not eval_scenario_dir.exists() or not list(eval_scenario_dir.glob("*.npz")):
            print(f"[ERROR] eval scenarios not found at {eval_scenario_dir.resolve()}")
            return 1
        print(f"[INFO] Train sources: {len(train_sources)}")
        for source in train_sources:
            print(f"[INFO]   {source['label']}: {source['file_count']} files cache={source['kf_cache_dir']}")
    else:
        for d, name in [(train_scenario_dir, "train"), (eval_scenario_dir, "eval")]:
            if not d.exists() or not list(d.glob("*.npz")):
                print(f"[ERROR] {name} scenarios not found at {d.resolve()}")
                return 1
        print(f"[INFO] Train: {len(list(train_scenario_dir.glob('*.npz')))} files")
    print(f"[INFO] Eval:  {len(list(eval_scenario_dir.glob('*.npz')))} files")

    run_dir, run_tag = resolve_scratch_run(config, args.seed)
    train_log_dir = run_dir / "logs" / "train"
    eval_log_dir = run_dir / "logs" / "eval"
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    write_run_metadata(
        run_dir=run_dir,
        config=config,
        seed=args.seed,
        config_source=pathlib.Path(args.config),
    )
    tensorboard_log_dir = None if args.no_tensorboard else train_log_dir / "tb"
    if tensorboard_log_dir is not None:
        os.makedirs(tensorboard_log_dir, exist_ok=True)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Seed: {args.seed}")

    n_envs = int(ppo_cfg.get("n_envs", 8))
    if train_sources and n_envs != len(train_sources):
        raise ValueError(f"ppo.n_envs must equal train_sources count ({len(train_sources)}) for equal-source mixed training")
    seed = args.seed
    n_eval_episodes = int(eval_cfg.get("n_eval_episodes", 100))

    if train_sources:
        train_env_kwargs_list = []
        for source in train_sources:
            source_config = dict(config)
            source_cache = dict(source_config.get("cache", {}))
            source_cache["kf_cache_dir"] = source["kf_cache_dir"]
            source_cache["kf_cache_mode"] = source["kf_cache_mode"]
            source_config["cache"] = source_cache
            kwargs = build_env_kwargs(source_config, source["scenario_dir"])
            kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
            train_env_kwargs_list.append(kwargs)
    else:
        train_env_kwargs = build_env_kwargs(config, train_scenario_dir)
        train_env_kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)
    eval_env_kwargs = build_env_kwargs(config, eval_scenario_dir)
    eval_env_kwargs["grid_map"] = np.zeros((32, 32), dtype=np.uint8)

    if train_sources:
        env = make_vec_env_from_kwargs_list(seed, USVEnvMinimal, train_env_kwargs_list, use_subproc=True)
    else:
        env = make_vec_env(n_envs, seed, USVEnvMinimal, train_env_kwargs, use_subproc=True)
    eval_env = make_vec_env(1, seed + 1000, USVEnvMinimal, eval_env_kwargs, use_subproc=True)

    norm_obs_keys = ["state", "dyn"]
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
        norm_obs_keys=norm_obs_keys,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
        gamma=0.99,
        norm_obs_keys=norm_obs_keys,
    )

    value_loss_type = resolve_value_loss_type(config)
    freeze_sigma = bool(critic_cfg.get("freeze_sigma", False))

    policy_kwargs = dict(
        features_extractor_class=MinimalFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=int(features_cfg.get("dim", 128))),
        value_loss_type=value_loss_type,
        freeze_sigma=freeze_sigma,
        log_std_init=-1.0,
    )

    n_steps = int(ppo_cfg.get("rollout_steps", 1024))
    batch_size = int(ppo_cfg.get("batch_size", 2048))
    n_epochs = int(ppo_cfg.get("epochs", 4))
    lr = float(ppo_cfg.get("learning_rate_initial", 3e-4))
    lr_final = float(ppo_cfg.get("learning_rate_final", 5e-5))
    lr_schedule_type = str(ppo_cfg.get("lr_schedule", "linear_floor"))

    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.15))
    ent_coef = float(ppo_cfg.get("entropy_coef", 0.02))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    target_kl = float(ppo_cfg.get("target_kl", 0.03))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

    if lr_schedule_type == "constant":
        lr_fn = constant_schedule(lr)
    elif lr_schedule_type == "linear":
        lr_fn = linear_schedule(lr)
    elif lr_schedule_type == "linear_floor":
        lr_fn = linear_floor_schedule(lr, lr_final)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule_type}")
    print(f"[INFO] LR schedule: {lr_schedule_type}, initial={lr}, final={lr_final}")

    model = UnifiedPPOTrainer(
        policy=UnifiedPolicy,
        env=env,
        value_loss_type=value_loss_type,
        cwvl_trust_power=float(critic_cfg.get("trust_power", 1.0)),
        cwvl_normalize_trust_weights=bool(critic_cfg.get("normalize_trust_weights", False)),
        learning_rate=lr_fn,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        target_kl=target_kl,
        max_grad_norm=max_grad_norm,
        tensorboard_log=None if tensorboard_log_dir is None else str(tensorboard_log_dir),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=str(ppo_cfg.get("device", "auto")),
        seed=seed,
    )

    eval_freq_steps = int(eval_cfg.get("eval_freq_steps", 20000))
    eval_freq = max(1, eval_freq_steps // n_envs)
    checkpoint_freq = max(1, 20000 // n_envs)

    print(f"[INFO] Eval every {eval_freq} calls = {eval_freq * n_envs} steps, {n_eval_episodes} episodes")
    print(f"[INFO] N envs: {n_envs}, n_steps per rollout: {n_steps}")

    sync_cb = SyncVecNormalizeCallback(eval_env, verbose=0)
    save_ckpt_cb = SaveCheckpointCallback(str(run_dir / "checkpoints"), checkpoint_freq, verbose=0)


    train_metrics_cb = TrainingMetricsCallback(
        csv_path=str(train_log_dir / "training_metrics.csv"),
        verbose=0,
    )

    callback = CallbackList([sync_cb, save_ckpt_cb, train_metrics_cb])

    if args.smoke:
        total_steps = 10000
    elif args.train_steps is not None:
        total_steps = args.train_steps
    else:
        total_steps = int(config.get("train_steps", 1000000))
    print(f"[INFO] Total training steps: {total_steps}")

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback,
            progress_bar=False,
            tb_log_name=None if args.no_tensorboard else run_tag,
        )

        final_model_path = str(run_dir / "final_model")
        model.save(final_model_path)
        env.save(str(run_dir / "vec_normalize_final.pkl"))
        print(f"[DONE] Model saved to {final_model_path}")

    except KeyboardInterrupt:
        print("[INTERRUPTED] Saving current model...")
        model.save(str(run_dir / "interrupted_model"))
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
