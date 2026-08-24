import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from simple_boat.envs.corecbf_lie import (
    CredibleGeometryDomainError,
    aggregate_colregs_duty,
    colregs_duty,
    colregs_reference_control,
    corecbf_terms,
    credibility_aware_corecbf_terms,
    credible_covariance,
    credible_geometry_scale,
)
from simple_boat.envs.usv_env_minimal import USVEnvMinimal
from scripts.eval_transfer import aggregate_step_latencies
from scripts.train_obs3 import build_env_kwargs


PARAMS = {
    "m11": 68.448727294,
    "m22": 155.0,
    "m33": 41.85,
    "x_u": -77.554432348,
    "y_v": -155.0,
    "n_r": -41.85,
    "yaw_nonlinear": 10.0,
}


def test_credible_covariance_adds_only_positive_semidefinite_mismatch():
    p_filter = np.diag([2.0, 1.0, 3.0, 4.0])
    p_tmse = np.diag([1.0, 4.0, 3.0, 2.0])

    result = credible_covariance(p_filter, p_tmse)

    np.testing.assert_allclose(result, np.diag([2.0, 4.0, 3.0, 4.0]))
    assert np.min(np.linalg.eigvalsh(result - p_filter)) >= -1e-12
    assert np.min(np.linalg.eigvalsh(result - p_tmse)) >= -1e-12


def test_credible_geometry_reduces_to_deterministic_geometry_without_uncertainty():
    result = credible_geometry_scale(
        relative_position=np.array([10.0, 0.0]),
        position_covariance=np.zeros((2, 2)),
        safety_distance=2.0,
        confidence_scale=2.0,
    )

    assert result["lambda"] == pytest.approx(1.0)
    assert result["alpha"] == pytest.approx(math.asin(0.2))


def test_credible_geometry_shrinks_lambda_when_lateral_uncertainty_grows():
    result = credible_geometry_scale(
        relative_position=np.array([10.0, 0.0]),
        position_covariance=np.diag([0.0, 1.0]),
        safety_distance=2.0,
        confidence_scale=1.0,
    )

    assert 0.0 < result["lambda"] < 1.0
    assert result["e_perp"] == pytest.approx(1.0)


def test_credible_geometry_rejects_states_outside_the_theoretical_domain():
    with pytest.raises(CredibleGeometryDomainError, match="credible geometry domain"):
        credible_geometry_scale(
            relative_position=np.array([2.1, 0.0]),
            position_covariance=np.diag([0.04, 0.0]),
            safety_distance=2.0,
            confidence_scale=1.0,
        )


def test_credible_geometry_does_not_relabel_invalid_covariance_as_domain_exit():
    with pytest.raises(ValueError, match="covariance semidefinite") as caught:
        credible_geometry_scale(
            relative_position=np.array([10.0, 0.0]),
            position_covariance=np.diag([-1.0, 1.0]),
            safety_distance=2.0,
            confidence_scale=1.0,
        )

    assert not isinstance(caught.value, CredibleGeometryDomainError)


def test_credibility_aware_terms_match_existing_corecbf_when_lambda_is_one():
    kwargs = dict(
        ship_state=np.array([0.0, 0.0, 0.2, 1.1, 0.1, -0.05]),
        target_position=np.array([8.0, 2.0]),
        target_velocity=np.array([-0.2, 0.1]),
        safety_distance=2.0,
        surge_accel=0.8,
        turn_accel=1.2,
        turn_direction=-1,
        usv_params=PARAMS,
    )

    baseline = corecbf_terms(**kwargs)
    credible = credibility_aware_corecbf_terms(**kwargs, geometry_scale=1.0)

    assert credible["H"] == pytest.approx(baseline["H"])
    assert credible["LfH"] == pytest.approx(baseline["LfH"])
    np.testing.assert_allclose(credible["LgH"], baseline["LgH"])


def test_credibility_aware_lie_derivative_matches_finite_difference():
    ship_state = np.array([0.0, 0.0, 0.2, 1.1, 0.1, -0.05])
    target_position = np.array([8.0, 2.0])
    target_velocity = np.array([-0.2, 0.1])
    control = np.array([15.0, -4.0])
    kwargs = dict(
        safety_distance=2.0,
        surge_accel=0.8,
        turn_accel=1.2,
        turn_direction=-1,
        usv_params=PARAMS,
        geometry_scale=0.4,
    )
    terms = credibility_aware_corecbf_terms(
        ship_state=ship_state,
        target_position=target_position,
        target_velocity=target_velocity,
        **kwargs,
    )

    _, _, psi, u, v, r = ship_state
    c, s = math.cos(psi), math.sin(psi)
    tau_u, tau_r = control
    derivative = np.array(
        [
            c * u - s * v,
            s * u + c * v,
            r,
            (tau_u + PARAMS["x_u"] * u + PARAMS["m22"] * v * r) / PARAMS["m11"],
            (PARAMS["y_v"] * v - PARAMS["m11"] * u * r) / PARAMS["m22"],
            (
                tau_r
                + PARAMS["n_r"] * (1.0 + PARAMS["yaw_nonlinear"] * abs(r)) * r
                - (PARAMS["m22"] - PARAMS["m11"]) * u * v
            )
            / PARAMS["m33"],
        ]
    )
    eps = 1e-6
    next_terms = credibility_aware_corecbf_terms(
        ship_state=ship_state + eps * derivative,
        target_position=target_position + eps * target_velocity,
        target_velocity=target_velocity,
        **kwargs,
    )
    finite_difference = (next_terms["H"] - terms["H"]) / eps

    assert finite_difference == pytest.approx(
        terms["LfH"] + float(terms["LgH"] @ control),
        rel=2e-5,
        abs=2e-5,
    )


def test_credibility_aware_barrier_is_negative_inside_physical_radius():
    terms = credibility_aware_corecbf_terms(
        ship_state=np.zeros(6),
        target_position=np.array([1.5, 0.0]),
        target_velocity=np.zeros(2),
        safety_distance=2.0,
        surge_accel=0.8,
        turn_accel=1.2,
        turn_direction=-1,
        usv_params=PARAMS,
        geometry_scale=0.4,
    )

    assert terms["H"] < 0.0


def test_colregs_duty_activates_for_negative_bearing_starboard_crossing():
    kwargs = dict(
        own_velocity=np.array([1.0, 0.0]),
        target_velocity=np.array([-1.0, 0.0]),
        d_safe=10.0,
        tcpa_horizon=10.0,
        kappa_beta=60.0,
        kappa_time=60.0,
        theta_head=math.radians(5.0),
        theta_limit=math.radians(112.5),
    )

    starboard = colregs_duty(relative_position=np.array([5.0, -3.0]), **kwargs)
    port = colregs_duty(relative_position=np.array([5.0, 3.0]), **kwargs)

    assert starboard["beta"] < 0.0
    assert starboard["phi"] > 0.5
    assert port["beta"] > 0.0
    assert port["phi"] < 1e-3


def test_colregs_duty_is_symmetric_across_head_on_bearing():
    kwargs = dict(
        own_velocity=np.array([1.0, 0.0]),
        target_velocity=np.array([-1.0, 0.0]),
        d_safe=3.0,
        tcpa_horizon=10.0,
        kappa_beta=60.0,
        kappa_time=60.0,
        theta_head=math.radians(5.0),
        theta_limit=math.radians(112.5),
    )

    right = colregs_duty(relative_position=np.array([5.0, 0.05]), **kwargs)
    left = colregs_duty(relative_position=np.array([5.0, -0.05]), **kwargs)

    assert right["psi_head"] == pytest.approx(left["psi_head"])


def test_colregs_aggregation_and_reference_control_follow_theory():
    duty = aggregate_colregs_duty([0.2, 0.5])
    reference = colregs_reference_control(
        nominal_control=np.array([10.0, 5.0]),
        aggregate_duty=duty,
        turn_accel=2.0,
        yaw_control_gain=0.25,
    )

    assert duty == pytest.approx(0.6)
    np.testing.assert_allclose(reference, np.array([10.0, 0.2]))


def test_colregs_reference_control_scales_bias_linearly_and_preserves_default():
    kwargs = dict(
        nominal_control=np.array([10.0, 5.0]),
        aggregate_duty=0.6,
        turn_accel=2.0,
        yaw_control_gain=0.25,
    )

    default = colregs_reference_control(**kwargs)
    full = colregs_reference_control(**kwargs, reference_scale=1.0)
    half = colregs_reference_control(**kwargs, reference_scale=0.5)
    off = colregs_reference_control(**kwargs, reference_scale=0.0)

    np.testing.assert_allclose(default, full)
    np.testing.assert_allclose(off, kwargs["nominal_control"])
    np.testing.assert_allclose(half, off + 0.5 * (full - off))


@pytest.mark.parametrize(
    "reference_scale",
    [-1e-12, 1.0 + 1e-12, np.nan, np.inf, -np.inf],
)
def test_colregs_reference_control_rejects_invalid_reference_scale(reference_scale):
    with pytest.raises(ValueError, match="invalid COLREGs reference parameters"):
        colregs_reference_control(
            nominal_control=np.array([10.0, 5.0]),
            aggregate_duty=0.6,
            turn_accel=2.0,
            yaw_control_gain=0.25,
            reference_scale=reference_scale,
        )


def test_usv_env_has_no_rc_colregs_mode_api():
    assert "rc_colregs_mode" not in inspect.signature(
        USVEnvMinimal.__init__
    ).parameters


def test_environment_defaults_to_standard_colregs_reference_scale():
    default = inspect.signature(USVEnvMinimal.__init__).parameters[
        "corecbf_colregs_reference_scale"
    ].default

    assert default == pytest.approx(0.1875)


def _env(variant: str, reference_scale: float = 1.0) -> USVEnvMinimal:
    return USVEnvMinimal(
        grid_map=np.zeros((64, 64), dtype=np.uint8),
        dynamic_obstacles=False,
        use_filter=True,
        civo_shield_enabled=True,
        civo_shield_distance=5.0,
        corecbf_variant=variant,
        corecbf_colregs_reference_scale=reference_scale,
        rc_colregs_enabled=True,
        rc_colregs_d_safe=30.0,
        rc_colregs_tau=100.0,
    )


def _set_single_target(
    env: USVEnvMinimal,
    *,
    estimate: np.ndarray,
    truth: np.ndarray,
    filter_covariance: np.ndarray | None = None,
) -> None:
    env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    env.dyn_step = 0
    env.dyn_traj = np.asarray(truth, dtype=float).reshape(1, 1, 4)
    env.dyn_hat_world = [np.asarray(estimate, dtype=float).reshape(1, 4)]
    covariance = np.zeros((4, 4)) if filter_covariance is None else filter_covariance
    env.dyn_P_world = [np.asarray(covariance, dtype=float).reshape(1, 4, 4)]
    env.obstacle_estimates = {
        0: {
            "dx": float(estimate[0]),
            "dy": float(estimate[1]),
            "vx": float(estimate[2]),
            "vy": float(estimate[3]),
            "radius": 1.0,
        }
    }
    env._refresh_civo_cones()
    env._update_colregs_signal()


def test_credible_colregs_environment_consumes_reference_scale():
    full = _env("credibility_colregs", reference_scale=1.0)
    half = _env("credibility_colregs", reference_scale=0.5)
    off = _env("credibility_colregs", reference_scale=0.0)
    try:
        target = np.array([20.0, -12.0, -1.0, 0.0])
        for env in (full, half, off):
            _set_single_target(env, estimate=target, truth=target)

        action = np.zeros(2)
        nominal = off._action_to_control(action)
        full_reference = full._corecbf_reference_control(action)
        half_reference = half._corecbf_reference_control(action)
        off_reference = off._corecbf_reference_control(action)

        assert full._last_colregs_aggregate_duty == pytest.approx(
            half._last_colregs_aggregate_duty
        )
        assert full._last_colregs_aggregate_duty == pytest.approx(
            off._last_colregs_aggregate_duty
        )
        np.testing.assert_allclose(off_reference, nominal)
        np.testing.assert_allclose(
            half_reference,
            off_reference + 0.5 * (full_reference - off_reference),
        )
    finally:
        full.close()
        half.close()
        off.close()


def test_deterministic_corecbf_ignores_reference_scale():
    full = _env("deterministic", reference_scale=1.0)
    off = _env("deterministic", reference_scale=0.0)
    try:
        action = np.array([0.2, -0.3])

        np.testing.assert_allclose(
            full._corecbf_reference_control(action),
            off._corecbf_reference_control(action),
        )
    finally:
        full.close()
        off.close()


@pytest.mark.parametrize(
    "reference_scale",
    [-0.1, 1.1, np.nan, np.inf, -np.inf],
)
def test_environment_rejects_invalid_colregs_reference_scale(reference_scale):
    with pytest.raises(
        ValueError,
        match="corecbf_colregs_reference_scale must be finite and lie in",
    ):
        _env("credibility_colregs", reference_scale=reference_scale)


def test_rc_colregs_keeps_continuous_starboard_reference():
    env = USVEnvMinimal(
        grid_map=np.zeros((64, 64), dtype=np.uint8),
        dynamic_obstacles=False,
        use_filter=True,
        civo_shield_enabled=True,
        civo_shield_distance=5.0,
        corecbf_variant="credibility_colregs",
        rc_colregs_enabled=True,
        rc_colregs_d_safe=30.0,
        rc_colregs_tau=100.0,
    )
    try:
        target = np.array([20.0, -12.0, -1.0, 0.0])
        _set_single_target(env, estimate=target, truth=target)

        nominal_control = env._action_to_control(np.zeros(2))
        reference = env._corecbf_reference_control(np.zeros(2))
        duty = env._last_colregs_aggregate_duty
        expected = colregs_reference_control(
            nominal_control=nominal_control,
            aggregate_duty=duty,
            turn_accel=env.corecbf_turn_accel,
            yaw_control_gain=1.0 / float(env.usv_params["m33"]),
            reference_scale=env.corecbf_colregs_reference_scale,
        )

        assert 0.0 < duty < 1.0
        np.testing.assert_allclose(reference, expected)
        assert nominal_control[1] > reference[1] > (
            nominal_control[1]
            - env.corecbf_turn_accel * float(env.usv_params["m33"])
        )
    finally:
        env.close()


def test_credible_variant_uses_oracle_tmse_to_scale_corecbf_geometry():
    credible_env = _env("credibility_colregs")
    deterministic_env = _env("deterministic")
    try:
        estimate = np.array([10.0, 0.0, -1.0, 0.0])
        truth = np.array([10.0, 2.0, -1.0, 0.0])
        _set_single_target(credible_env, estimate=estimate, truth=truth)
        _set_single_target(deterministic_env, estimate=estimate, truth=truth)

        credible = credible_env._civo_corecbf_terms(0)
        deterministic = deterministic_env._civo_corecbf_terms(0)

        assert 0.0 < credible["geometry_scale"] < 1.0
        assert credible_env.civo_cones[0]["half_angle"] == pytest.approx(
            credible["credible_alpha"]
        )
        assert deterministic["geometry_scale"] == pytest.approx(1.0)
    finally:
        credible_env.close()
        deterministic_env.close()


def test_domain_exit_falls_back_to_deterministic_terms_then_recovers():
    env = _env("credibility_colregs")
    deterministic = _env("deterministic")
    try:
        env.corecbf_safety_distance = 2.2
        deterministic.corecbf_safety_distance = 2.2
        estimate = np.array([5.971362, 0.0, -1.0, 0.0])
        truth = estimate.copy()
        _set_single_target(
            env,
            estimate=estimate,
            truth=truth,
            filter_covariance=np.eye(4),
        )
        _set_single_target(deterministic, estimate=estimate, truth=truth)
        env.dyn_traj = np.repeat(truth.reshape(1, 1, 4), 2, axis=0)
        env.dyn_hat_world = [np.repeat(estimate.reshape(1, 4), 2, axis=0)]
        env.dyn_P_world = [np.stack([np.eye(4), 0.01 * np.eye(4)])]

        env.dyn_step = 0
        fallback = env._civo_corecbf_terms(0)
        baseline = deterministic._civo_corecbf_terms(0)
        assert fallback["credible_geometry_fallback"] is True
        assert fallback["H"] == pytest.approx(baseline["H"])
        assert fallback["LfH"] == pytest.approx(baseline["LfH"])
        np.testing.assert_allclose(fallback["LgH"], baseline["LgH"])

        env.dyn_step = 1
        recovered = env._civo_corecbf_terms(0)
        assert recovered["credible_geometry_fallback"] is False
        assert 0.0 < recovered["geometry_scale"] <= 1.0
    finally:
        env.close()
        deterministic.close()


def test_mixed_targets_share_one_qp_while_only_domain_exit_target_falls_back():
    env = _env("credibility_colregs")
    try:
        env.corecbf_safety_distance = 2.2
        estimates = np.array(
            [[5.971362, 0.0, -1.0, 0.0], [10.0, 2.0, -1.0, 0.0]],
            dtype=float,
        )
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.dyn_step = 0
        env.dyn_traj = estimates.reshape(1, 2, 4)
        env.dyn_hat_world = [row.reshape(1, 4) for row in estimates]
        env.dyn_P_world = [np.eye(4).reshape(1, 4, 4), np.zeros((1, 4, 4))]
        env.obstacle_estimates = {
            obs_id: {
                "dx": float(row[0]),
                "dy": float(row[1]),
                "vx": float(row[2]),
                "vy": float(row[3]),
                "radius": 1.0,
            }
            for obs_id, row in enumerate(estimates)
        }

        assert env._civo_corecbf_terms(0)["credible_geometry_fallback"] is True
        assert env._civo_corecbf_terms(1)["credible_geometry_fallback"] is False
        solved = env._solve_civo_corecbf_qp(np.zeros(2), (0, 1))

        assert solved["status"]
        assert env._last_credible_geometry_fallback_ids == (0,)
        assert env._credible_geometry_fallback_steps == 1
        assert env._credible_geometry_fallback_obstacle_steps == 1
    finally:
        env.close()


def test_invalid_covariance_is_not_hidden_by_environment_fallback():
    env = _env("credibility_colregs")
    try:
        estimate = np.array([10.0, 0.0, -1.0, 0.0])
        _set_single_target(env, estimate=estimate, truth=estimate)
        env.dyn_P_world = [np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(1, 4, 4)]

        with pytest.raises(ValueError, match="positive semidefinite"):
            env._civo_corecbf_terms(0)
    finally:
        env.close()


def test_credible_colregs_reference_bias_activates_only_for_starboard_crossing():
    starboard_env = _env("credibility_colregs")
    port_env = _env("credibility_colregs")
    deterministic_env = _env("deterministic")
    try:
        starboard = np.array([20.0, -12.0, -1.0, 0.0])
        port = np.array([20.0, 12.0, -1.0, 0.0])
        _set_single_target(starboard_env, estimate=starboard, truth=starboard)
        _set_single_target(port_env, estimate=port, truth=port)
        _set_single_target(deterministic_env, estimate=starboard, truth=starboard)

        starboard_action = starboard_env._shield_action_civo(np.zeros(2))
        port_action = port_env._shield_action_civo(np.zeros(2))
        deterministic_action = deterministic_env._shield_action_civo(np.zeros(2))

        assert starboard_env._last_colregs_aggregate_duty > 0.0
        assert starboard_action[1] < 0.0
        assert port_env._last_colregs_aggregate_duty < 1e-3
        assert (
            starboard_env._last_colregs_aggregate_duty
            > 1_000.0 * port_env._last_colregs_aggregate_duty
        )
        assert abs(float(starboard_action[1])) > 10.0 * abs(float(port_action[1]))
        np.testing.assert_allclose(deterministic_action, np.zeros(2), atol=1e-6)
    finally:
        starboard_env.close()
        port_env.close()
        deterministic_env.close()


def test_training_config_selects_new_variant_without_changing_default():
    default_kwargs = build_env_kwargs({}, Path("scenarios"))
    credible_kwargs = build_env_kwargs(
        {"civo": {"corecbf_variant": "credibility_colregs"}},
        Path("scenarios"),
    )

    assert default_kwargs["corecbf_variant"] == "deterministic"
    assert credible_kwargs["corecbf_variant"] == "credibility_colregs"


def test_eval_summary_counts_credible_geometry_fallback_events():
    rows = [
        {
            "civo_shield_latency_ms": 1.0,
            "policy_predict_latency_ms": 2.0,
            "step_wall_time_ms": 3.0,
            "credible_geometry_fallback_count": 0,
        },
        {
            "civo_shield_latency_ms": 1.0,
            "policy_predict_latency_ms": 2.0,
            "step_wall_time_ms": 3.0,
            "credible_geometry_fallback_count": 2,
        },
    ]

    summary = aggregate_step_latencies(rows)

    assert summary["credible_geometry_fallback_step_count"] == 1
    assert summary["credible_geometry_fallback_obstacle_step_count"] == 2
