import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simple_boat.envs.usv_env_minimal as usv_env_module
from simple_boat.envs.relaxed_vo_cbf import (
    RelaxedVOCBFDomainError,
    RelaxedVOCBFNoVerifiedAction,
    hard_collision_barrier_value,
    hard_collision_cbf_terms,
    otter_acceleration_maps,
    relaxed_vo_cbf_terms,
    solve_relaxed_affine_qp,
    solve_relaxed_vo_cbf_qp,
    time_to_collision,
    vo_barrier_value,
)
from simple_boat.envs.usv_env_minimal import SafetyFilterRuntimeError, USVEnvMinimal
from scripts.train_obs3 import build_env_kwargs


PARAMS = {
    "m11": 68.448727294,
    "m22": 155.0,
    "m33": 41.85,
    "x_u": -77.554432348,
    "y_v": -155.0,
    "n_r": -41.85,
    "yaw_nonlinear": 10.0,
    "y_pontoon": 0.395,
}


def acceleration_maps() -> tuple[np.ndarray, np.ndarray, float, float]:
    return otter_acceleration_maps(
        m11=PARAMS["m11"],
        m33=PARAMS["m33"],
        y_p=PARAMS["y_pontoon"],
        thrust_lower=-66.708,
        thrust_upper=119.682,
    )


def test_otter_acceleration_maps_round_trip_and_scales():
    thrust_to_control, control_to_thrust, surge_scale, yaw_scale = (
        acceleration_maps()
    )
    expected_surge_scale = min(
        2.0 * 119.682 / PARAMS["m11"],
        2.0 * 66.708 / PARAMS["m11"],
    )
    expected_yaw_scale = (
        PARAMS["y_pontoon"] * (119.682 + 66.708) / PARAMS["m33"]
    )
    thrusts = np.array([40.0, -20.0])
    control = thrust_to_control @ thrusts

    assert surge_scale == pytest.approx(expected_surge_scale)
    assert yaw_scale == pytest.approx(expected_yaw_scale)
    assert control[0] == pytest.approx(20.0 / PARAMS["m11"])
    assert control[1] == pytest.approx(
        (surge_scale / yaw_scale)
        * (PARAMS["y_pontoon"] * 60.0 / PARAMS["m33"])
    )
    np.testing.assert_allclose(
        thrust_to_control @ control_to_thrust, np.eye(2), atol=1e-12
    )
    np.testing.assert_allclose(
        control_to_thrust @ thrust_to_control, np.eye(2), atol=1e-12
    )


def ship_derivative(ship_state: np.ndarray, control: np.ndarray) -> np.ndarray:
    _, _, psi, u, v, r = ship_state
    c, s = math.cos(psi), math.sin(psi)
    tau_u, tau_r = control
    return np.array(
        [
            c * u - s * v,
            s * u + c * v,
            r,
            (tau_u + PARAMS["x_u"] * u + PARAMS["m22"] * v * r)
            / PARAMS["m11"],
            (PARAMS["y_v"] * v - PARAMS["m11"] * u * r) / PARAMS["m22"],
            (
                tau_r
                + PARAMS["n_r"]
                * (1.0 + PARAMS["yaw_nonlinear"] * abs(r))
                * r
                - (PARAMS["m22"] - PARAMS["m11"]) * u * v
            )
            / PARAMS["m33"],
        ],
        dtype=float,
    )


def test_time_to_collision_uses_smallest_positive_disk_root():
    assert time_to_collision([5.0, 0.0], [-1.0, 0.0], 2.0) == pytest.approx(
        3.0
    )
    assert math.isinf(time_to_collision([5.0, 0.0], [1.0, 0.0], 2.0))
    assert math.isinf(time_to_collision([5.0, 0.0], [0.0, 1.0], 2.0))
    assert math.isinf(time_to_collision([5.0, 0.0], [0.0, 0.0], 2.0))


def test_fixed_state_matches_paper_vo_formula_and_author_slack_orientation():
    terms = relaxed_vo_cbf_terms(
        ship_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        target_position=np.array([5.0, 0.0]),
        target_velocity=np.zeros(2),
        safety_distance=2.0,
        usv_params=PARAMS,
    )

    assert terms is not None
    assert terms["H"] == pytest.approx(-5.0 + math.sqrt(21.0))
    assert terms["ttc"] == pytest.approx(3.0)
    np.testing.assert_allclose(
        terms["LgH_thrust"], [terms["LgH_tau"][0]] * 2
    )


def test_pure_barrier_values_match_existing_direct_terms():
    ship = np.array([0.0, 0.0, 0.3, 1.0, 0.2, 0.1])
    target_position = np.array([5.0, 2.0])
    target_velocity = np.array([-0.1, 0.05])
    vo = relaxed_vo_cbf_terms(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
        usv_params=PARAMS,
    )
    hard = hard_collision_cbf_terms(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
        braking_accel=1.0,
        usv_params=PARAMS,
    )

    assert vo is not None
    assert hard is not None
    assert vo_barrier_value(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
    ) == pytest.approx(vo["H"])
    assert hard_collision_barrier_value(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
        braking_accel=1.0,
    ) == pytest.approx(hard["H"])


def test_affine_core_matches_direct_wrapper():
    common = dict(
        nominal_thrusts=np.array([0.2, -0.1]),
        control_to_thrust=np.eye(2),
        vo_ttc=np.array([2.0]),
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
    )
    direct = solve_relaxed_vo_cbf_qp(
        vo_h=np.array([0.2]),
        vo_lf_h=np.array([-0.4]),
        vo_lg_thrust=np.array([[0.3, -0.2]]),
        hard_h=np.array([0.4]),
        hard_lf_h=np.array([-0.1]),
        hard_lg_thrust=np.array([[0.1, 0.2]]),
        alpha_vo=10.0,
        alpha_c=10.0,
        **common,
    )
    affine = solve_relaxed_affine_qp(
        vo_offset=np.array([1.6]),
        vo_input_rows=np.array([[0.3, -0.2]]),
        vo_slack_ids=np.array([0]),
        hard_offset=np.array([3.9]),
        hard_input_rows=np.array([[0.1, 0.2]]),
        **common,
    )

    np.testing.assert_allclose(affine["thrusts"], direct["thrusts"])
    np.testing.assert_allclose(affine["slacks"], direct["slacks"])


def test_affine_core_shares_one_slack_across_prediction_stages():
    solved = solve_relaxed_affine_qp(
        nominal_thrusts=np.zeros(2),
        control_to_thrust=np.eye(2),
        vo_offset=np.array([-1.0, -2.0]),
        vo_input_rows=np.zeros((2, 2)),
        vo_slack_ids=np.array([0, 0]),
        vo_ttc=np.array([1.0]),
        hard_offset=np.empty(0),
        hard_input_rows=np.empty((0, 2)),
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
    )

    assert solved["slacks"].shape == (1,)
    assert solved["slacks"][0] == pytest.approx(2.0, abs=2e-5)
    np.testing.assert_allclose(
        solved["relaxed_vo_residuals"], [1.0, 0.0], atol=2e-5
    )


def test_affine_core_enforces_total_thrust_bounds():
    solved = solve_relaxed_affine_qp(
        nominal_thrusts=np.array([2.0, 2.0]),
        control_to_thrust=np.eye(2),
        vo_offset=np.empty(0),
        vo_input_rows=np.empty((0, 2)),
        vo_slack_ids=np.empty(0, dtype=int),
        vo_ttc=np.empty(0),
        hard_offset=np.empty(0),
        hard_input_rows=np.empty((0, 2)),
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        thrust_sum_lower=-1.0,
        thrust_sum_upper=1.0,
    )

    assert float(np.sum(solved["thrusts"])) == pytest.approx(1.0, abs=2e-5)
    np.testing.assert_allclose(solved["thrusts"], [0.5, 0.5], atol=2e-5)


def test_relaxed_vo_cbf_terms_have_surge_only_control_row():
    terms = relaxed_vo_cbf_terms(
        ship_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        target_position=np.array([5.0, 0.0]),
        target_velocity=np.zeros(2),
        safety_distance=2.0,
        usv_params=PARAMS,
    )
    assert terms is not None
    _, control_to_thrust, _, _ = acceleration_maps()
    control_row = terms["LgH_thrust"] @ control_to_thrust

    assert abs(control_row[0]) > 1e-12
    assert control_row[1] == pytest.approx(0.0, abs=1e-12)


def test_vo_and_hard_cbf_lie_derivatives_match_finite_differences():
    ship = np.array([0.5, -0.2, 0.3, 1.0, 0.2, 0.1])
    target_position = np.array([5.0, 2.0])
    target_velocity = np.array([-0.2, -0.05])
    control = np.array([10.0, 2.0])
    eps = 1e-6
    ship_dot = ship_derivative(ship, control)
    braking_accel = 1.9491377747

    vo = relaxed_vo_cbf_terms(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
        usv_params=PARAMS,
    )
    hard = hard_collision_cbf_terms(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=2.0,
        braking_accel=braking_accel,
        usv_params=PARAMS,
    )
    assert vo is not None
    assert hard is not None

    def values(s: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        vo_terms = relaxed_vo_cbf_terms(
            ship_state=s,
            target_position=p,
            target_velocity=target_velocity,
            safety_distance=2.0,
            usv_params=PARAMS,
        )
        hard_terms = hard_collision_cbf_terms(
            ship_state=s,
            target_position=p,
            target_velocity=target_velocity,
            safety_distance=2.0,
            braking_accel=braking_accel,
            usv_params=PARAMS,
        )
        assert vo_terms is not None
        assert hard_terms is not None
        return float(vo_terms["H"]), float(hard_terms["H"])

    plus = values(
        ship + eps * ship_dot, target_position + eps * target_velocity
    )
    minus = values(
        ship - eps * ship_dot, target_position - eps * target_velocity
    )
    numerical = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    analytic = np.array(
        [
            vo["LfH"] + vo["LgH_tau"] @ control,
            hard["LfH"] + hard["LgH_tau"] @ control,
        ]
    )
    np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-6)


def test_vo_domain_exit_does_not_remove_closing_hard_cbf():
    common = dict(
        ship_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        target_velocity=np.zeros(2),
        safety_distance=2.2,
        usv_params=PARAMS,
    )
    with pytest.raises(
        RelaxedVOCBFDomainError, match="distance > safety_distance"
    ):
        relaxed_vo_cbf_terms(
            target_position=np.array([2.1, 0.0]), **common
        )

    hard = hard_collision_cbf_terms(
        target_position=np.array([2.1, 0.0]),
        braking_accel=1.9491377747,
        **common,
    )
    assert hard is not None
    assert hard["distance"] == pytest.approx(2.1)
    assert hard["closing_speed"] == pytest.approx(1.0)
    assert np.all(np.isfinite(hard["LgH_thrust"]))


def test_hard_cbf_rejects_only_zero_distance():
    common = dict(
        ship_state=np.zeros(6),
        target_velocity=np.zeros(2),
        safety_distance=2.2,
        braking_accel=1.0,
        usv_params=PARAMS,
    )
    with pytest.raises(ValueError, match="positive distance"):
        hard_collision_cbf_terms(
            target_position=np.zeros(2),
            **common,
        )


def test_side_on_closing_geometry_exposes_zero_first_order_authority():
    hard = hard_collision_cbf_terms(
        ship_state=np.zeros(6),
        target_position=np.array([0.0, 5.0]),
        target_velocity=np.array([0.0, -1.0]),
        safety_distance=2.0,
        braking_accel=1.0,
        usv_params=PARAMS,
    )
    assert hard is not None
    assert hard["closing_speed"] == pytest.approx(1.0)
    np.testing.assert_allclose(hard["LgH_thrust"], 0.0, atol=1e-12)


def empty_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.empty(0), np.empty(0), np.empty((0, 2))


def test_qp_uses_one_nonnegative_slack_per_vo_row_with_author_sign():
    hard_h, hard_lf, hard_lg = empty_rows()
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.zeros(2),
        vo_h=np.zeros(2),
        vo_lf_h=np.array([-1.0, -2.0]),
        vo_lg_thrust=np.zeros((2, 2)),
        vo_ttc=np.array([1.0, 2.0]),
        hard_h=hard_h,
        hard_lf_h=hard_lf,
        hard_lg_thrust=hard_lg,
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        control_to_thrust=np.eye(2),
    )

    np.testing.assert_allclose(solved["slacks"], [1.0, 2.0], atol=2e-5)
    np.testing.assert_allclose(
        solved["relaxed_vo_residuals"], [0.0, 0.0], atol=2e-5
    )
    assert solved["hard_residuals"].size == 0


def test_hard_row_has_no_slack_and_infeasibility_is_not_hidden():
    vo_h, vo_lf, vo_lg = empty_rows()
    with pytest.raises(RuntimeError, match="infeasible|failed"):
        solve_relaxed_vo_cbf_qp(
            nominal_thrusts=np.zeros(2),
            vo_h=vo_h,
            vo_lf_h=vo_lf,
            vo_lg_thrust=vo_lg,
            vo_ttc=np.empty(0),
            hard_h=np.zeros(1),
            hard_lf_h=-np.ones(1),
            hard_lg_thrust=np.zeros((1, 2)),
            alpha_vo=10.0,
            alpha_c=10.0,
            k_u=1.0,
            k_vo=1000.0,
            thrust_lower=-2.0,
            thrust_upper=3.0,
            control_to_thrust=np.eye(2),
        )


def test_qp_fast_path_bounds_and_rank_diagnostics():
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.array([0.25, -0.10]),
        vo_h=np.ones(1),
        vo_lf_h=np.zeros(1),
        vo_lg_thrust=np.array([[0.5, 0.5]]),
        vo_ttc=np.ones(1),
        hard_h=np.ones(1),
        hard_lf_h=np.zeros(1),
        hard_lg_thrust=np.array([[-0.2, -0.2]]),
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        control_to_thrust=np.eye(2),
    )

    assert solved["status"] == "nominal feasible"
    assert solved["solver"] == "none"
    np.testing.assert_allclose(solved["thrusts"], [0.25, -0.10])
    assert solved["safety_row_rank"] == 1
    assert solved["yaw_sensitivity_max"] == pytest.approx(0.5)


def test_qp_projects_only_to_exact_thrust_box_and_preserves_differential_command():
    vo_h, vo_lf, vo_lg = empty_rows()
    hard_h, hard_lf, hard_lg = empty_rows()
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.array([10.0, -10.0]),
        vo_h=vo_h,
        vo_lf_h=vo_lf,
        vo_lg_thrust=vo_lg,
        vo_ttc=np.empty(0),
        hard_h=hard_h,
        hard_lf_h=hard_lf,
        hard_lg_thrust=hard_lg,
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        control_to_thrust=np.eye(2),
    )

    np.testing.assert_allclose(solved["thrusts"], [3.0, -2.0], atol=2e-5)
    assert np.all(solved["thrusts"] >= -2.0)
    assert np.all(solved["thrusts"] <= 3.0)


def test_qp_scaled_eta_matches_ttc_weighted_lambda_solution():
    hard_h, hard_lf, hard_lg = empty_rows()
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.zeros(2),
        vo_h=np.zeros(1),
        vo_lf_h=-np.ones(1),
        vo_lg_thrust=np.array([[1.0, 0.0]]),
        vo_ttc=np.array([2.0]),
        hard_h=hard_h,
        hard_lf_h=hard_lf,
        hard_lg_thrust=hard_lg,
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=2.0,
        control_to_thrust=np.eye(2),
    )
    weight = 500.0
    expected_lambda = 1.0 / (1.0 + weight)

    assert solved["control"][0] == pytest.approx(
        weight / (1.0 + weight), abs=2e-5
    )
    assert solved["slacks"][0] == pytest.approx(
        expected_lambda, abs=2e-5
    )
    assert solved["eta_slacks"][0] == pytest.approx(
        math.sqrt(weight) * expected_lambda, abs=2e-5
    )
    assert solved["control"][1] == pytest.approx(0.0, abs=2e-5)


def test_qp_enforces_exact_propeller_box_in_acceleration_coordinates():
    vo_h, vo_lf, vo_lg = empty_rows()
    hard_h, hard_lf, hard_lg = empty_rows()
    _, control_to_thrust, _, _ = acceleration_maps()
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.array([200.0, -150.0]),
        vo_h=vo_h,
        vo_lf_h=vo_lf,
        vo_lg_thrust=vo_lg,
        vo_ttc=np.empty(0),
        hard_h=hard_h,
        hard_lf_h=hard_lf,
        hard_lg_thrust=hard_lg,
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-66.708,
        thrust_upper=119.682,
        control_to_thrust=control_to_thrust,
    )
    actual_thrusts = control_to_thrust @ solved["control"]

    np.testing.assert_allclose(actual_thrusts, solved["thrusts"], atol=2e-5)
    assert np.all(actual_thrusts >= -66.708 - 2e-5)
    assert np.all(actual_thrusts <= 119.682 + 2e-5)


def test_qp_surge_only_row_preserves_nominal_yaw_control():
    vo_h, vo_lf, vo_lg = empty_rows()
    solved = solve_relaxed_vo_cbf_qp(
        nominal_thrusts=np.array([0.0, 0.4]),
        vo_h=vo_h,
        vo_lf_h=vo_lf,
        vo_lg_thrust=vo_lg,
        vo_ttc=np.empty(0),
        hard_h=np.zeros(1),
        hard_lf_h=-np.ones(1),
        hard_lg_thrust=np.array([[1.0, 0.0]]),
        alpha_vo=10.0,
        alpha_c=10.0,
        k_u=1.0,
        k_vo=1000.0,
        thrust_lower=-2.0,
        thrust_upper=2.0,
        control_to_thrust=np.eye(2),
    )

    np.testing.assert_allclose(solved["control"], [1.0, 0.4], atol=2e-5)
    assert solved["yaw_sensitivity_max"] == pytest.approx(0.0)


def test_solver_candidate_requires_optimized_status_and_verified_residual(
    monkeypatch,
):
    import osqp

    vo_h, vo_lf, vo_lg = empty_rows()

    def solve():
        return solve_relaxed_vo_cbf_qp(
            nominal_thrusts=np.zeros(2),
            vo_h=vo_h,
            vo_lf_h=vo_lf,
            vo_lg_thrust=vo_lg,
            vo_ttc=np.empty(0),
            hard_h=np.zeros(1),
            hard_lf_h=-np.ones(1),
            hard_lg_thrust=np.array([[1.0, 0.0]]),
            alpha_vo=10.0,
            alpha_c=10.0,
            k_u=1.0,
            k_vo=1000.0,
            thrust_lower=-2.0,
            thrust_upper=2.0,
            control_to_thrust=np.eye(2),
        )

    def result(status, candidate):
        return SimpleNamespace(
            info=SimpleNamespace(status=status, iter=1),
            x=np.asarray(candidate, dtype=float),
        )

    monkeypatch.setattr(
        osqp.OSQP,
        "solve",
        lambda self, raise_error=False: result(
            "solved inaccurate", [1.0, 0.0]
        ),
    )
    np.testing.assert_allclose(solve()["control"], [1.0, 0.0])

    monkeypatch.setattr(
        osqp.OSQP,
        "solve",
        lambda self, raise_error=False: result(
            "maximum iterations reached", [1.0, 0.0]
        ),
    )
    with pytest.raises(RelaxedVOCBFNoVerifiedAction, match="status"):
        solve()

    monkeypatch.setattr(
        osqp.OSQP,
        "solve",
        lambda self, raise_error=False: result("solved", [0.0, 0.0]),
    )
    with pytest.raises(RelaxedVOCBFNoVerifiedAction, match="residual"):
        solve()


def test_training_config_maps_relaxed_vo_settings_without_changing_default():
    default = build_env_kwargs({}, Path("scenarios"))
    configured = build_env_kwargs(
        {
            "civo": {
                "shield_enabled": True,
                "shield_method": "relaxed_vo_cbf",
                "safety_distance": 2.2,
                "vo_cbf_alpha_vo": 100.0,
                "vo_cbf_alpha_c": 10.0,
                "vo_cbf_k_u": 1.0,
                "vo_cbf_k_vo": 10000.0,
            }
        },
        Path("scenarios"),
    )

    assert default["civo_shield_method"] == "corecbf"
    assert default["civo_shield_gate_mode"] == "distance_tcpa"
    assert "vo_cbf_prediction_steps" not in default
    assert configured["civo_shield_method"] == "relaxed_vo_cbf"
    assert configured["corecbf_safety_distance"] == 2.2
    assert configured["vo_cbf_alpha_vo"] == 100.0
    assert configured["vo_cbf_alpha_c"] == 10.0
    assert configured["vo_cbf_k_u"] == 1.0
    assert configured["vo_cbf_k_vo"] == 10000.0
    assert "vo_cbf_prediction_steps" not in configured

    with pytest.raises(ValueError, match="fixed to one step"):
        build_env_kwargs(
            {"civo": {"vo_cbf_prediction_steps": 2}}, Path("scenarios")
        )


def test_environment_cbf_vo_uses_calibrated_kvo_without_an_override():
    parameters = inspect.signature(USVEnvMinimal.__init__).parameters

    assert parameters["vo_cbf_k_u"].default == 2.0
    assert parameters["vo_cbf_k_vo"].default == 50.0


def test_training_cbf_vo_uses_calibrated_kvo_without_an_override():
    defaults = build_env_kwargs({}, Path("scenarios"))

    assert defaults["vo_cbf_k_u"] == 2.0
    assert defaults["vo_cbf_k_vo"] == 50.0


def make_direct_relaxed_vo_cbf_env() -> USVEnvMinimal:
    return USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method="relaxed_vo_cbf",
        corecbf_safety_distance=2.2,
    )


def make_cbf_vo_env() -> USVEnvMinimal:
    return USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method="cbf_vo",
        corecbf_safety_distance=2.2,
    )


def test_cbf_vo_accepts_tau_u_rate_limit_inside_filter():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method="cbf_vo",
        corecbf_safety_distance=2.2,
        actuator_tau_u_dot_max=50.0,
    )
    try:
        assert env.actuator_tau_u_dot_max == pytest.approx(50.0)
    finally:
        env.close()


def test_non_predictive_cbf_vo_filters_still_reject_tau_u_rate_limit():
    with pytest.raises(ValueError, match="safety filters do not include"):
        USVEnvMinimal(
            grid_map=np.zeros((32, 32), dtype=np.uint8),
            dynamic_obstacles=False,
            civo_shield_enabled=True,
            civo_shield_method="relaxed_vo_cbf",
            corecbf_safety_distance=2.2,
            actuator_tau_u_dot_max=50.0,
        )


def test_cbf_vo_does_not_expose_prediction_steps_setting():
    env = make_cbf_vo_env()
    try:
        assert not hasattr(env, "vo_cbf_prediction_steps")
    finally:
        env.close()


def test_cbf_vo_rejects_retired_prediction_steps_argument():
    with pytest.raises(TypeError, match="fixed to one step"):
        USVEnvMinimal(
            grid_map=np.zeros((32, 32), dtype=np.uint8),
            dynamic_obstacles=False,
            vo_cbf_prediction_steps=2,
        )


def test_predictive_relaxed_vo_cbf_method_identifier_is_retired():
    with pytest.raises(ValueError, match="civo_shield_method"):
        USVEnvMinimal(
            grid_map=np.zeros((32, 32), dtype=np.uint8),
            dynamic_obstacles=False,
            civo_shield_method="predictive_relaxed_vo_cbf",
        )


def configure_cbf_vo_test_obstacles(env: USVEnvMinimal) -> None:
    env.ship_state = np.array([0.0, 0.0, 0.15, 1.2, 0.1, 0.05])
    env.obstacle_estimates = {
        0: {
            "dx": 4.0,
            "dy": 1.0,
            "vx": -0.2,
            "vy": 0.0,
            "radius": 1.0,
        },
        1: {
            "dx": 5.0,
            "dy": -1.5,
            "vx": -0.1,
            "vy": 0.1,
            "radius": 1.0,
        },
    }
    env.civo_cones = {}


@pytest.mark.parametrize(
    ("method", "mode", "expected"),
    [
        ("corecbf", "native", (0,)),
        ("cbf_vo", "native", (0, 1)),
        ("corecbf", "all_obstacles", (0, 1)),
        ("cbf_vo", "distance_tcpa", (0,)),
    ],
)
def test_shield_gate_mode_selects_expected_obstacles(method, mode, expected):
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method=method,
        civo_shield_gate_mode=mode,
    )
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {"dx": 5.0, "dy": 0.0, "vx": 0.0, "vy": 0.0},
            1: {"dx": 20.0, "dy": 0.0, "vx": 0.0, "vy": 0.0},
        }

        assert env._civo_shield_obstacle_ids() == expected
    finally:
        env.close()


def test_shield_gate_defaults_to_distance_tcpa():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
    )
    try:
        assert env.civo_shield_gate_mode == "distance_tcpa"
    finally:
        env.close()


def test_cbf_vo_rows_respect_explicit_distance_tcpa_gate():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method="cbf_vo",
        civo_shield_gate_mode="distance_tcpa",
        corecbf_safety_distance=2.2,
    )
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {"dx": 5.0, "dy": 1.0, "vx": 0.0, "vy": 0.0},
            1: {"dx": 20.0, "dy": 1.0, "vx": 0.0, "vy": 0.0},
        }

        rows = env._cbf_vo_rows(np.zeros(2))

        assert tuple(rows["targets"]) == (0,)
    finally:
        env.close()


def test_shield_gate_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="civo_shield_gate_mode"):
        USVEnvMinimal(
            grid_map=np.zeros((32, 32), dtype=np.uint8),
            dynamic_obstacles=False,
            civo_shield_gate_mode="sometimes",
        )


def test_generic_two_step_thrust_rollout_matches_full_model_twice():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
    )
    try:
        thrusts = np.array([20.0, -5.0])
        before = env.ship_state.copy()
        predicted = env._predict_states_for_thrusts(thrusts, steps=2)
        n_port, n_stbd = env._thrusts_to_prop_speeds(thrusts)
        first = usv_env_module.update_usv_full_model(
            before.copy(), n_port, n_stbd, env.dt, env.usv_params
        )
        first[2] = usv_env_module.normalize_angle_0_to_2pi(first[2])
        second = usv_env_module.update_usv_full_model(
            first, n_port, n_stbd, env.dt, env.usv_params
        )
        second[2] = usv_env_module.normalize_angle_0_to_2pi(second[2])

        np.testing.assert_allclose(predicted, np.vstack((first, second)))
        np.testing.assert_array_equal(env.ship_state, before)
    finally:
        env.close()


def test_cbf_vo_rows_have_stable_yaw_authority_and_rank_two():
    env = make_cbf_vo_env()
    try:
        configure_cbf_vo_test_obstacles(env)
        rows = env._cbf_vo_rows(np.zeros(2))

        assert rows["yaw_sensitivity_max"] > 1e-8
        assert rows["safety_row_rank"] == 2
        np.testing.assert_allclose(
            rows["input_rows"],
            rows["half_epsilon_input_rows"],
            rtol=5e-2,
            atol=1e-7,
        )
    finally:
        env.close()


def test_cbf_vo_rows_use_only_stage_one():
    env = make_cbf_vo_env()
    try:
        configure_cbf_vo_test_obstacles(env)
        rows = env._cbf_vo_rows(np.zeros(2))

        assert {stage for _, stage in rows["hard_row_keys"]} == {1}
        assert {stage for _, stage in rows["vo_row_keys"]} <= {1}
    finally:
        env.close()


def test_cbf_vo_row_rollouts_use_one_step(monkeypatch):
    env = make_cbf_vo_env()
    original = env._predict_states_for_thrusts
    calls = []

    def predict(thrusts, *, steps):
        calls.append(steps)
        return original(thrusts, steps=steps)

    monkeypatch.setattr(env, "_predict_states_for_thrusts", predict)
    try:
        configure_cbf_vo_test_obstacles(env)
        env._cbf_vo_rows(np.zeros(2))

        assert calls
        assert set(calls) == {1}
    finally:
        env.close()


def test_cbf_vo_solver_rejects_affine_safe_nonlinear_hard_violation(
    monkeypatch,
):
    env = make_cbf_vo_env()
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {
                "dx": 2.4,
                "dy": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "radius": 1.0,
            }
        }

        def false_affine_safe(**kwargs):
            return {
                "thrusts": np.full(2, kwargs["thrust_upper"]),
                "slacks": np.zeros(kwargs["vo_ttc"].size),
                "vo_residuals": np.ones(kwargs["vo_offset"].size),
                "relaxed_vo_residuals": np.ones(kwargs["vo_offset"].size),
                "hard_residuals": np.ones(kwargs["hard_offset"].size),
                "status": "synthetic affine-safe",
                "solver": "synthetic",
                "iterations": 1,
                "safety_row_rank": 2,
                "yaw_sensitivity_max": 1.0,
            }

        monkeypatch.setattr(
            usv_env_module, "solve_relaxed_affine_qp", false_affine_safe
        )

        with pytest.raises(
            RelaxedVOCBFNoVerifiedAction, match="nonlinear predictive"
        ):
            env._solve_cbf_vo_qp(np.zeros(2))
    finally:
        env.close()


def test_cbf_vo_nonlinear_verification_rolls_one_step(monkeypatch):
    env = make_cbf_vo_env()
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    target = {
        "position": np.array([20.0, 0.0]),
        "velocity": np.zeros(2),
        "vo_h": None,
        "hard_h": hard_collision_barrier_value(
            ship_state=state,
            target_position=np.array([20.0, 0.0]),
            target_velocity=np.zeros(2),
            safety_distance=env.corecbf_safety_distance,
            braking_accel=1.0,
        ),
    }
    calls = []

    monkeypatch.setattr(
        env,
        "_cbf_vo_rows",
        lambda action: {
            "nominal_thrusts": np.zeros(2),
            "control_to_thrust": np.eye(2),
            "vo_offset": np.empty(0),
            "vo_input_rows": np.empty((0, 2)),
            "vo_slack_ids": np.empty(0, dtype=int),
            "vo_ttc": np.empty(0),
            "hard_offset": np.zeros(1),
            "hard_input_rows": np.zeros((1, 2)),
            "vo_row_keys": (),
            "hard_row_keys": ((0, 1),),
            "vo_domain_exit_keys": (),
            "vo_current_domain_exit_ids": (),
            "targets": {0: target},
            "braking_accel": 1.0,
        },
    )
    monkeypatch.setattr(
        usv_env_module,
        "solve_relaxed_affine_qp",
        lambda **kwargs: {
            "thrusts": np.zeros(2),
            "slacks": np.empty(0),
            "status": "synthetic",
            "solver": "synthetic",
            "iterations": 1,
            "safety_row_rank": 1,
            "yaw_sensitivity_max": 0.0,
        },
    )
    monkeypatch.setattr(
        env,
        "_predict_states_for_thrusts",
        lambda thrusts, *, steps: (
            calls.append(steps) or np.repeat(state[None, :], steps, axis=0)
        ),
    )
    try:
        env._solve_cbf_vo_qp(np.zeros(2))

        assert calls == [1]
    finally:
        env.close()


def test_cbf_vo_method_dispatches_and_reports_prediction_diagnostics(
    monkeypatch,
):
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_method="cbf_vo",
        corecbf_safety_distance=2.2,
    )
    calls = 0

    def solve(action):
        nonlocal calls
        calls += 1
        nominal = env._action_to_thrusts(action)
        return {
            "thrusts": nominal,
            "nominal_thrusts": nominal,
            "slacks": np.empty(0),
            "relaxed_vo_residuals": np.empty(0),
            "hard_residuals": np.empty(0),
            "vo_h": np.empty(0),
            "hard_h": np.empty(0),
            "vo_obstacle_ids": (),
            "hard_obstacle_ids": (),
            "status": "synthetic predictive",
            "solver": "synthetic",
            "iterations": 1,
            "safety_row_rank": 2,
            "yaw_sensitivity_max": 0.25,
            "predictive_residual_error_max": 0.03,
            "vo_domain_exit_keys": (),
        }

    monkeypatch.setattr(
        env, "_solve_cbf_vo_qp", solve
    )
    try:
        filtered = env._shield_action_civo(np.zeros(2, dtype=np.float32))

        assert filtered.shape == (2,)
        assert calls == 1
        assert env._last_civo_mechanism["cbf_vo_prediction_steps"] == 1
        assert env._last_civo_mechanism[
            "vo_cbf_predictive_residual_error_max"
        ] == pytest.approx(0.03)
    finally:
        env.close()


def test_cbf_vo_row_diagnostics_count_rows_before_fallback(monkeypatch):
    env = make_cbf_vo_env()
    try:
        configure_cbf_vo_test_obstacles(env)
        rows = env._cbf_vo_rows(np.zeros(2))
        monkeypatch.setattr(
            usv_env_module,
            "solve_relaxed_affine_qp",
            lambda **kwargs: (_ for _ in ()).throw(
                RelaxedVOCBFNoVerifiedAction("synthetic solver failure")
            ),
        )

        env._shield_action_civo(np.zeros(2, dtype=np.float32))

        assert env._last_civo_mechanism["vo_cbf_fallback"] == 1
        assert env._last_civo_mechanism["vo_cbf_active_vo_row_count"] == len(
            rows["vo_row_keys"]
        )
        assert env._last_civo_mechanism["vo_cbf_active_hard_row_count"] == len(
            rows["hard_row_keys"]
        )
    finally:
        env.close()


def test_environment_falls_back_once_and_retries_next_step(monkeypatch):
    env = make_direct_relaxed_vo_cbf_env()
    original = usv_env_module.solve_relaxed_vo_cbf_qp
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RelaxedVOCBFNoVerifiedAction(
                "synthetic infeasible hard row"
            )
        return original(**kwargs)

    monkeypatch.setattr(usv_env_module, "solve_relaxed_vo_cbf_qp", fail_once)
    try:
        env.obstacle_estimates = {}
        nominal = np.array([0.4, 0.1], dtype=np.float32)

        assert np.array_equal(env._shield_action_civo(nominal), nominal)
        assert env._last_civo_qp_success is False
        assert env._last_civo_mechanism["vo_cbf_fallback"] == 1
        assert env._vo_cbf_ep_fallback_count == 1

        assert env._shield_action_civo(nominal).shape == (2,)
        assert calls == 2
        assert env._last_civo_qp_success is True
        assert env._last_civo_mechanism["vo_cbf_fallback"] == 0
        assert env._vo_cbf_ep_fallback_count == 1
    finally:
        env.close()


def test_environment_does_not_fallback_on_program_error(monkeypatch):
    env = make_direct_relaxed_vo_cbf_env()

    def fail_with_shape_bug(**kwargs):
        raise ValueError("synthetic shape bug")

    monkeypatch.setattr(
        usv_env_module, "solve_relaxed_vo_cbf_qp", fail_with_shape_bug
    )
    try:
        with pytest.raises(SafetyFilterRuntimeError, match="synthetic shape bug"):
            env._shield_action_civo(np.zeros(2, dtype=np.float32))

        assert env._vo_cbf_ep_fallback_count == 0
    finally:
        env.close()


def test_environment_baseline_uses_all_estimates_and_exact_propeller_override():
    env = make_direct_relaxed_vo_cbf_env()
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {
                "dx": 4.0,
                "dy": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "radius": 1.0,
            },
            1: {
                "dx": 6.0,
                "dy": 0.5,
                "vx": -0.2,
                "vy": 0.0,
                "radius": 1.0,
            },
        }
        env.civo_cones = {}

        filtered = env._shield_action_civo(
            np.array([0.7, 0.2], dtype=np.float32)
        )

        assert filtered.shape == (2,)
        assert env._last_civo_prop_override is not None
        assert env._last_civo_qp_success is True
        assert env._last_civo_constraint_count >= 2
        assert env._last_civo_mechanism["vo_cbf_active_vo_count"] == 2
        assert env._last_civo_mechanism["vo_cbf_safety_row_rank"] <= 1
        assert env._last_civo_mechanism[
            "vo_cbf_yaw_sensitivity_max"
        ] == pytest.approx(0.0)
    finally:
        env.close()


def test_environment_skips_vo_domain_row_but_keeps_hard_row():
    env = make_direct_relaxed_vo_cbf_env()
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {
                "dx": 2.1,
                "dy": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "radius": 1.0,
            }
        }

        with pytest.raises(
            RelaxedVOCBFNoVerifiedAction
        ):
            env._solve_civo_relaxed_vo_cbf_qp(np.zeros(2))

        assert env._last_civo_mechanism["vo_cbf_domain_exit_count"] == 1
        assert env._last_civo_mechanism["vo_cbf_active_vo_count"] == 0
        assert env._last_civo_mechanism["vo_cbf_active_hard_count"] == 1
    finally:
        env.close()
