import math
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_dqn_vo_baseline import (
    algorithm_signature,
    env_config_for_dqn_vo,
    load_resume_state,
    resolve_requested_steps,
    resolve_train_sources,
    save_resume_bundle,
    validate_resume_state,
)
from scripts.eval_dqn_vo_baseline import aggregate_control_latency
from simple_boat.envs.dqn_vo_baseline import (
    DQNVOObservationRewardWrapper,
    risk_score,
    starboard_required,
    tcpa_dcpa,
    velocity_inside_vo,
)
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def test_control_latency_is_weighted_by_actual_steps():
    summary = aggregate_control_latency(
        [
            {"steps": 2, "control_latency_total_ms": 4.0},
            {"steps": 3, "control_latency_total_ms": 12.0},
        ]
    )

    assert summary == {"control_latency_steps": 5, "control_latency_mean_ms": 3.2}


def test_discrete_actions_map_to_fixed_surge_and_three_yaw_commands():
    env = USVEnvMinimal(grid_map=np.zeros((32, 32), dtype=np.uint8), dynamic_obstacles=False)
    wrapped = DQNVOObservationRewardWrapper(env, fixed_surge=0.7, yaw_command=0.45)
    try:
        assert np.allclose(wrapped.map_action(0), [0.7, 0.45])
        assert np.allclose(wrapped.map_action(1), [0.7, 0.0])
        assert np.allclose(wrapped.map_action(2), [0.7, -0.45])
    finally:
        wrapped.close()


def test_tcpa_dcpa_identifies_approaching_close_pass():
    tcpa, dcpa = tcpa_dcpa(dx=10.0, dy=0.0, own_vx=1.0, own_vy=0.0, obs_vx=0.0, obs_vy=0.0)

    assert math.isclose(tcpa, 10.0)
    assert math.isclose(dcpa, 0.0)
    assert risk_score(tcpa=tcpa, dcpa=dcpa, horizon=12.0, warning_distance=4.0) > 0.5


def test_drl_vo_helpers_are_owned_by_baseline_module():
    assert tcpa_dcpa.__module__ == "simple_boat.envs.dqn_vo_baseline"
    assert risk_score.__module__ == "simple_boat.envs.dqn_vo_baseline"
    assert starboard_required.__module__ == "simple_boat.envs.dqn_vo_baseline"


def test_velocity_inside_vo_uses_relative_velocity_cone():
    assert velocity_inside_vo(
        dx=10.0,
        dy=0.0,
        own_vx=1.0,
        own_vy=0.0,
        obs_vx=0.0,
        obs_vy=0.0,
        radius=2.0,
    )
    assert not velocity_inside_vo(
        dx=10.0,
        dy=0.0,
        own_vx=-1.0,
        own_vy=0.0,
        obs_vx=0.0,
        obs_vy=0.0,
        radius=2.0,
    )


def test_starboard_required_for_head_on_or_starboard_crossing_sector():
    boundary = math.radians(22.5)

    assert starboard_required(dx=10.0, dy=0.0)
    assert starboard_required(dx=1.0, dy=math.tan(boundary))
    assert starboard_required(dx=8.0, dy=-4.0)
    assert not starboard_required(dx=8.0, dy=4.0)
    assert not starboard_required(dx=-1.0, dy=-1.0)


def test_dqn_vo_selects_the_highest_risk_target():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
    )
    wrapped = DQNVOObservationRewardWrapper(env)
    try:
        env.ship_state[3:5] = [1.0, 0.0]
        env.obstacle_estimates = {
            4: {"dx": 8.0, "dy": 0.0, "vx": 0.0, "vy": 0.0},
            9: {"dx": 4.0, "dy": 0.0, "vx": 0.0, "vy": 0.0},
            2: {"dx": 10.0, "dy": 0.0, "vx": 0.0, "vy": 0.0},
        }

        selected = wrapped._select_target()

        assert selected["dx"] == 4.0
        assert selected["risk"] > 0.0
    finally:
        wrapped.close()


def test_train_sources_preserve_per_obs_cache_mapping():
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
        root = pathlib.Path(tmp)
        obs3 = root / "nav3_new_map"
        obs4 = root / "nav4_new_map"
        obs3.mkdir()
        obs4.mkdir()
        for i in range(2):
            (obs3 / f"obs3_{i}.npz").touch()
        for i in range(3):
            (obs4 / f"obs4_{i}.npz").touch()

        sources = resolve_train_sources(
            {
                "data": {
                    "train_sources": [
                        {"label": "obs3", "scenario_dir": str(obs3), "kf_cache_dir": "cache/obs3"},
                        {"label": "obs4", "scenario_dir": str(obs4), "kf_cache_dir": "cache/obs4"},
                    ],
                },
            },
        )

    assert [s["label"] for s in sources] == ["obs3", "obs4"]
    assert [s["file_count"] for s in sources] == [2, 3]
    assert [s["kf_cache_dir"] for s in sources] == ["cache/obs3", "cache/obs4"]


def test_dqn_vo_env_config_keeps_uncertain_perception_and_disables_only_civo():
    cfg = {"env": {"use_filter": True}, "cache": {"kf_cache_mode": "off"}}

    resolved = env_config_for_dqn_vo(cfg, "results/kf_cache/obs3_cache")

    assert resolved["env"]["use_filter"] is True
    assert resolved["cache"] == {
        "kf_cache_dir": "results/kf_cache/obs3_cache",
        "kf_cache_mode": "read_strict",
    }
    assert resolved["civo"] == {"enabled": False, "shield_enabled": False}
    assert resolved["rc_colregs"] == {"enabled": False, "reward_weight": 0.0}


def test_dqn_vo_timing_config_recomputes_filter_online():
    resolved = env_config_for_dqn_vo(
        {"env": {"use_filter": True}},
        "results/kf_cache/obs3_cache",
        online_exact=True,
    )

    assert resolved["env"]["filter_execution_mode"] == "online_exact"
    assert resolved["cache"]["kf_cache_mode"] == "read_strict"


def test_requested_steps_accepts_cli_override_without_mutating_config():
    config = {"dqn": {"max_timesteps": 5_000_000, "smoke_timesteps": 20_000}}

    assert resolve_requested_steps(config, smoke=False, override=123_456) == 123_456
    assert resolve_requested_steps(config, smoke=True, override=256) == 256
    assert config["dqn"]["max_timesteps"] == 5_000_000


def test_resume_state_rejects_seed_or_algorithm_signature_mismatch():
    config = {
        "dqn_vo": {"fixed_surge": 0.7, "yaw_command": 0.45},
        "dqn": {
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "buffer_size": 1_000_000,
            "batch_size": 1024,
            "target_update_interval": 10_000,
            "exploration_final_eps": 0.1,
            "learning_starts": 10_000,
        },
    }
    signature = algorithm_signature(config)
    state = {"schema_version": 1, "seed": 2, "steps": 500_000, "algorithm_signature": signature}

    validate_resume_state(state, seed=2, max_steps=5_000_000, signature=signature)
    with np.testing.assert_raises_regex(ValueError, "seed"):
        validate_resume_state(state, seed=3, max_steps=5_000_000, signature=signature)
    with np.testing.assert_raises_regex(ValueError, "hyperparameters"):
        validate_resume_state(state, seed=2, max_steps=5_000_000, signature={**signature, "gamma": 0.5})
    with np.testing.assert_raises_regex(ValueError, "max_steps"):
        validate_resume_state(state, seed=2, max_steps=100_000, signature=signature)


def test_resume_bundle_saves_model_replay_normalization_and_metadata(tmp_path):
    class FakeVecNormalize:
        def save(self, path):
            pathlib.Path(path).write_bytes(b"vec")

    class FakeEnv:
        def env_method(self, name):
            assert name == "get_resume_state"
            return [{"scenario": 1}, {"scenario": 2}]

    class FakeModel:
        num_timesteps = 500_000

        def save(self, path):
            pathlib.Path(path).write_bytes(b"model")

        def save_replay_buffer(self, path):
            pathlib.Path(path).write_bytes(b"replay")

        def get_vec_normalize_env(self):
            return FakeVecNormalize()

        def get_env(self):
            return FakeEnv()

    state_path = save_resume_bundle(
        FakeModel(),
        run_dir=tmp_path,
        seed=3,
        signature={"gamma": 0.99},
        source_signature=(("obs3", "maps", "cache", 10),),
    )

    state = load_resume_state(state_path)
    assert state["seed"] == 3
    assert state["steps"] == 500_000
    assert state["env_states"] == [{"scenario": 1}, {"scenario": 2}]
    assert pathlib.Path(state["model"]).read_bytes() == b"model"
    assert pathlib.Path(state["replay_buffer"]).read_bytes() == b"replay"
    assert pathlib.Path(state["vecnormalize"]).read_bytes() == b"vec"


