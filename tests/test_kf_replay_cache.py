import json

import numpy as np
import pytest

from scripts.eval_helpers import select_single_scenario_file
from scripts.precompute_kf_cache import (
    build_precompute_env_kwargs,
    cache_inventory,
    validate_cache_inventory,
    write_cache_audit,
)
from scripts.train_obs3 import build_env_kwargs
from simple_boat.envs.usv_env_minimal import (
    OnlineFilterCacheMismatch,
    USVEnvMinimal,
)


def _write_tiny_scenario(path, obstacle_count=1):
    grid = np.zeros((32, 32), dtype=np.uint8)
    dyn_traj = np.zeros((32, obstacle_count, 4), dtype=np.float32)
    dyn_traj[:, :, :2] = [20.0, 20.0]
    for k in range(5, 15):
        dyn_traj[k, 0, :2] = [float(k), 0.0]
    np.savez(
        path,
        grid=grid,
        init_pos=np.array([0.0, 0.0], dtype=np.float32),
        init_psi=np.float32(0.0),
        goal=np.array([31.0, 0.0], dtype=np.float32),
        dyn_traj=dyn_traj,
        dyn_seeds=np.arange(1234, 1234 + obstacle_count, dtype=np.int32),
    )


def _make_env(
    scenario_dir,
    cache_dir,
    cache_mode,
    filter_execution_mode="precomputed",
):
    return USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        scenario_dir=scenario_dir,
        load_on_reset=True,
        max_episode_steps=24,
        dt=0.1,
        dynamic_obstacles=True,
        use_filter=True,
        burst_enabled=True,
        burst_episode_prob=1.0,
        burst_start_mode="path_progress_random_candidate",
        burst_duration_steps=8,
        measurement_cov_scale=100.0,
        measurement_delay_steps=2,
        bias_enabled=False,
        path_progress_arrival_steps=31,
        risk_distance_threshold=0.5,
        trust_mode="oracle_tmse",
        kf_cache_dir=cache_dir,
        kf_cache_mode=cache_mode,
        filter_execution_mode=filter_execution_mode,
    )


def _load_tiny_scenario_manually(env, scenario_path):
    with np.load(scenario_path, allow_pickle=True) as data:
        env.set_grid(data["grid"])
        env._current_scenario_path = scenario_path
        env.fixed_initial_position = data["init_pos"]
        env.fixed_initial_psi = float(data["init_psi"])
        env.fixed_goal = data["goal"]
        env.set_dyn_traj(data["dyn_traj"])
        env.dyn_seeds = data["dyn_seeds"]


def test_set_dyn_traj_rejects_more_obstacles_than_default_capacity():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=True,
    )
    try:
        with pytest.raises(ValueError) as exc_info:
            env.set_dyn_traj(np.zeros((4, 10, 4), dtype=np.float32))
        message = str(exc_info.value)
        assert "10" in message
        assert "capacity 6" in message
        assert "construct env with scenario directory" in message
    finally:
        env.close()


def test_set_dyn_traj_normalizes_single_obstacle_track():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=True,
    )
    try:
        env.set_dyn_traj(np.zeros((4, 4), dtype=np.float64))
        assert env.dyn_traj.shape == (4, 1, 4)
        assert env.dyn_traj.dtype == np.float32
        assert env.dyn_obs_num == 1
    finally:
        env.close()


def test_build_precompute_env_kwargs_keeps_scenario_directory(tmp_path):
    scenario_dir = tmp_path / "maps"
    kwargs = build_precompute_env_kwargs({}, tmp_path / "cache", "write", scenario_dir)

    assert kwargs["scenario_dir"] == scenario_dir
    assert kwargs["load_on_reset"] is False


def test_build_precompute_env_kwargs_disables_online_exact_mode(tmp_path):
    kwargs = build_precompute_env_kwargs(
        {"env": {"filter_execution_mode": "online_exact"}},
        tmp_path / "cache",
        "write",
        tmp_path / "maps",
    )

    assert kwargs["filter_execution_mode"] == "precomputed"


def test_write_cache_audit_writes_stable_json(tmp_path):
    path = tmp_path / "audit.json"

    write_cache_audit(path, {"resets": 3, "cache_hits": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "cache_hits": 3,
        "resets": 3,
    }


def test_write_cache_audit_refuses_overwrite(tmp_path):
    path = tmp_path / "audit.json"
    write_cache_audit(path, {"status": "pass"})
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_cache_audit(path, {"status": "different"})

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_inventory_counts_recursive_npz_files_and_bytes(tmp_path):
    cache_dir = tmp_path / "cache"
    nested = cache_dir / "fingerprint"
    nested.mkdir(parents=True)
    first = nested / "first.npz"
    second = cache_dir / "second.npz"
    first.write_bytes(b"abc")
    second.write_bytes(b"12345")
    (nested / "ignored.json").write_text("{}", encoding="utf-8")

    assert cache_inventory(cache_dir) == {
        "cache_file_count": 2,
        "cache_bytes": 8,
        "temporary_files": [],
    }


def test_cache_inventory_rejects_stale_temporary_file(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "fingerprint.npz.tmp").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="temporary cache files"):
        cache_inventory(cache_dir)


def test_validate_cache_inventory_allows_non_binned_cache_reuse():
    validate_cache_inventory(
        {
            "cache_file_count": 1,
            "cache_bytes": 3,
            "temporary_files": [],
        },
        files_seen=30,
        resets=30,
        cache_hits=30,
        cache_mode="read_strict",
        all_binned_candidates=False,
    )


def test_validate_cache_inventory_rejects_zero_cache_bytes():
    with pytest.raises(RuntimeError, match="cache inventory is empty"):
        validate_cache_inventory(
            {
                "cache_file_count": 1,
                "cache_bytes": 0,
                "temporary_files": [],
            },
            files_seen=1,
            resets=1,
            cache_hits=1,
            cache_mode="read_strict",
            all_binned_candidates=True,
        )


def test_build_env_kwargs_maps_filter_execution_mode(tmp_path):
    kwargs = build_env_kwargs(
        {"env": {"filter_execution_mode": "online_exact"}},
        tmp_path,
    )

    assert kwargs["filter_execution_mode"] == "online_exact"


def test_obs10_scenario_directory_initializes_all_filter_tracks(tmp_path):
    scenario_dir = tmp_path / "maps"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "obs10.npz", obstacle_count=10)

    env = _make_env(scenario_dir, tmp_path / "cache", "off")
    try:
        obs, _ = env.reset(seed=0)
        assert env.dyn_obs_num == 10
        assert len(env.dyn_hat_world) == 10
        assert env.DYN_MAX == 10
        assert env.POLICY_DYN_SLOTS == 6
        assert env.observation_space["dyn"].shape == (42,)
        assert env.observation_space["dyn_mask"].shape == (6,)
        assert obs["dyn"].shape == (42,)
        assert obs["dyn_mask"].shape == (6,)
    finally:
        env.close()


def test_reset_propagates_obstacle_capacity_mismatch(tmp_path):
    scenario_path = tmp_path / "obs10.npz"
    _write_tiny_scenario(scenario_path, obstacle_count=10)
    env = _make_env(None, tmp_path / "cache", "off")
    select_single_scenario_file(env, scenario_path)

    try:
        with pytest.raises(ValueError, match=r"obstacle count 10.*capacity 6"):
            env.reset(seed=0)
    finally:
        env.close()


def test_kf_replay_cache_write_then_read_skips_filter_recompute(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    writer = _make_env(scenario_dir, cache_dir, "write")
    writer.reset(seed=0)
    cached_hat = writer.dyn_hat_world[0].copy()
    writer.close()

    reader = _make_env(scenario_dir, cache_dir, "read")

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("KF replay was recomputed instead of loaded from cache")

    reader._run_filter_with_burst = fail_if_recomputed
    reader.reset(seed=0)

    assert np.allclose(reader.dyn_hat_world[0], cached_hat)
    assert reader._kf_cache_hit
    reader.close()


def test_kf_replay_cache_read_strict_skips_filter_recompute(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    writer = _make_env(scenario_dir, cache_dir, "write")
    writer.reset(seed=0)
    cached_hat = writer.dyn_hat_world[0].copy()
    writer.close()

    reader = _make_env(scenario_dir, cache_dir, "read_strict")

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("KF replay was recomputed instead of loaded from cache")

    reader._run_filter_with_burst = fail_if_recomputed
    reader.reset(seed=0)

    assert np.allclose(reader.dyn_hat_world[0], cached_hat)
    assert reader._kf_cache_hit
    reader.close()


def test_kf_replay_cache_read_strict_can_search_multiple_roots(tmp_path):
    scenario_dir = tmp_path / "maps"
    missing_cache_dir = tmp_path / "missing_cache"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    writer = _make_env(scenario_dir, cache_dir, "write")
    writer.reset(seed=0)
    cached_hat = writer.dyn_hat_world[0].copy()
    writer.close()

    reader = _make_env(scenario_dir, [missing_cache_dir, cache_dir], "read_strict")

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("KF replay was recomputed instead of loaded from cache")

    reader._run_filter_with_burst = fail_if_recomputed
    reader.reset(seed=0)

    assert np.allclose(reader.dyn_hat_world[0], cached_hat)
    assert reader._kf_cache_hit
    assert cache_dir in reader._kf_cache_path.parents
    reader.close()


def test_kf_replay_cache_read_strict_raises_on_missing_entry(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    env = _make_env(scenario_dir, cache_dir, "read_strict")
    try:
        with pytest.raises(RuntimeError, match="Strict KF replay cache miss"):
            env.reset(seed=0)
    finally:
        env.close()


def test_kf_replay_cache_read_strict_requires_cache_dir():
    with pytest.raises(ValueError, match="read_strict requires kf_cache_dir"):
        _make_env(None, None, "read_strict")


def test_kf_replay_cache_rebuilds_when_forced_burst_changes_without_reload(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "tiny_000.npz"
    _write_tiny_scenario(scenario_path)

    env = _make_env(None, cache_dir, "write")
    _load_tiny_scenario_manually(env, scenario_path)

    env.forced_burst_start_step = 5
    env.reset(seed=0)
    first_path = env._kf_cache_path

    env.forced_burst_start_step = 6
    env.reset(seed=0)
    second_path = env._kf_cache_path

    assert first_path != second_path
    assert len(list(cache_dir.rglob("*.npz"))) == 2
    env.close()


def test_kf_replay_cache_write_permission_error_is_nonfatal(tmp_path, monkeypatch):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    original_replace = type(tmp_path).replace

    def deny_cache_replace(self, target):
        if str(target).endswith(".npz"):
            raise PermissionError("simulated concurrent cache writer")
        return original_replace(self, target)

    monkeypatch.setattr(type(tmp_path), "replace", deny_cache_replace)

    env = _make_env(scenario_dir, cache_dir, "write")
    try:
        env.reset(seed=0)
        assert len(env.dyn_hat_world) == 1
        assert env.dyn_hat_world[0].shape[0] == env.scenario_T
    finally:
        env.close()


def test_stateful_filter_steps_equal_full_trajectory(tmp_path):
    scenario_dir = tmp_path / "maps"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "tiny_000.npz"
    _write_tiny_scenario(scenario_path)
    env = _make_env(scenario_dir, tmp_path / "cache", "off")
    try:
        _load_tiny_scenario_manually(env, scenario_path)
        env.reset(seed=0)
        gt = env.dyn_traj[:, 0, :4].astype(float)
        env.trust_computer.reset()
        expected_hat, expected_tk, expected_p = env._run_filter_with_burst(
            gt, int(env.dyn_seeds[0]), obs_id=0
        )
        env.trust_computer.reset()
        actual_hat, actual_tk, actual_p = env._run_filter_statefully(
            gt, int(env.dyn_seeds[0]), obs_id=0
        )
        np.testing.assert_array_equal(actual_hat, expected_hat)
        np.testing.assert_array_equal(actual_tk, expected_tk)
        np.testing.assert_array_equal(actual_p, expected_p)
    finally:
        env.close()


def test_online_exact_mode_matches_every_cache_row(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(
        scenario_dir / "tiny_000.npz", obstacle_count=10
    )
    writer = _make_env(scenario_dir, cache_dir, "write")
    try:
        writer.reset(seed=0)
        expected_ids = tuple(writer._last_policy_obstacle_ids)
    finally:
        writer.close()

    reader = _make_env(
        scenario_dir,
        cache_dir,
        "read_strict",
        filter_execution_mode="online_exact",
    )
    try:
        reader.reset(seed=0)
        assert reader._kf_cache_hit
        assert reader._online_cache_checks == 10
        assert tuple(reader._last_policy_obstacle_ids) == expected_ids
        while reader.dyn_step + 1 < reader.scenario_T:
            reader.dyn_step += 1
            reader.dyn_pos = reader.dyn_traj[reader.dyn_step].copy()
            reader._filter_dynamic_obstacles()
            reader._get_obs()
            reader._assert_online_cache_step(reader.dyn_step)
        assert reader._online_cache_mismatches == 0
        assert reader._online_cache_checks == 10 * reader.scenario_T
    finally:
        reader.close()


def test_online_exact_reports_first_cache_mismatch(tmp_path):
    scenario_dir = tmp_path / "maps"
    cache_dir = tmp_path / "cache"
    scenario_dir.mkdir()
    _write_tiny_scenario(scenario_dir / "tiny_000.npz")
    writer = _make_env(scenario_dir, cache_dir, "write")
    try:
        writer.reset(seed=0)
    finally:
        writer.close()

    reader = _make_env(
        scenario_dir,
        cache_dir,
        "read_strict",
        filter_execution_mode="online_exact",
    )
    try:
        reader.reset(seed=0)
        reader._kf_reference_hat[0, 1, 0] += np.float32(1.0)
        reader.dyn_step = 1
        reader.dyn_pos = reader.dyn_traj[1].copy()
        reader._filter_dynamic_obstacles()
        reader._get_obs()
        with pytest.raises(
            OnlineFilterCacheMismatch,
            match=(
                r"map=tiny_000\.npz seed=0 anchor=.* step=1 "
                r"obstacle=0 field=dyn_hat_world"
            ),
        ):
            reader._assert_online_cache_step(1)
        assert reader._online_cache_mismatches == 1
    finally:
        reader.close()


def test_online_exact_requires_strict_cache(tmp_path):
    with pytest.raises(
        ValueError, match="online_exact requires kf_cache_mode='read_strict'"
    ):
        _make_env(
            None,
            tmp_path / "cache",
            "read",
            filter_execution_mode="online_exact",
        )


def test_online_exact_is_independent_of_future_ground_truth(tmp_path):
    cutoff = 8
    scenario_dirs = [tmp_path / "base", tmp_path / "changed"]
    for scenario_dir in scenario_dirs:
        scenario_dir.mkdir()
        _write_tiny_scenario(scenario_dir / "tiny_000.npz")

    changed_path = scenario_dirs[1] / "tiny_000.npz"
    with np.load(changed_path, allow_pickle=False) as data:
        changed = {key: data[key].copy() for key in data.files}
    changed["dyn_traj"][cutoff + 1 :, 0, 0] += np.float32(7.0)
    np.savez(changed_path, **changed)

    prefixes = []
    for index, scenario_dir in enumerate(scenario_dirs):
        cache_dir = tmp_path / f"cache_{index}"
        writer = _make_env(scenario_dir, cache_dir, "write")
        try:
            writer.forced_burst_start_step = 5
            writer.reset(seed=0)
        finally:
            writer.close()

        reader = _make_env(
            scenario_dir,
            cache_dir,
            "read_strict",
            filter_execution_mode="online_exact",
        )
        try:
            reader.forced_burst_start_step = 5
            reader.reset(seed=0)
            while reader.dyn_step < cutoff:
                reader.dyn_step += 1
                reader.dyn_pos = reader.dyn_traj[reader.dyn_step].copy()
                reader._filter_dynamic_obstacles()
                reader._get_obs()
                reader._assert_online_cache_step(reader.dyn_step)
            prefixes.append((
                np.asarray(reader.dyn_hat_world, dtype=np.float32),
                np.asarray(reader.dyn_tk, dtype=np.float32),
                np.asarray(reader.dyn_P_world, dtype=np.float32),
            ))
        finally:
            reader.close()

    for base, future_changed in zip(prefixes[0], prefixes[1]):
        np.testing.assert_array_equal(base, future_changed)
