import math
from pathlib import Path

import numpy as np
import pytest

from simple_boat.envs.corecbf_lie import corecbf_terms, solve_corecbf_qp
from simple_boat.envs.usv_env_minimal import USVEnvMinimal
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


def _ship_derivative(ship_state: np.ndarray, control: np.ndarray) -> np.ndarray:
    _, _, psi, u, v, r = ship_state
    c, s = math.cos(psi), math.sin(psi)
    tau_u, tau_r = control
    return np.array(
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
        ],
        dtype=float,
    )


def test_corecbf_lie_derivative_matches_finite_difference():
    ship = np.array([0.5, -0.2, 0.3, 1.0, 0.2, 0.1])
    target_position = np.array([5.0, 2.0])
    target_velocity = np.array([0.1, -0.05])
    control = np.array([10.0, 2.0])
    kwargs = dict(
        safety_distance=2.0,
        surge_accel=0.8,
        turn_accel=1.2,
        turn_direction=-1,
        usv_params=PARAMS,
    )

    terms = corecbf_terms(
        ship_state=ship,
        target_position=target_position,
        target_velocity=target_velocity,
        **kwargs,
    )
    eps = 1e-6
    ship_dot = _ship_derivative(ship, control)
    h_plus = corecbf_terms(
        ship_state=ship + eps * ship_dot,
        target_position=target_position + eps * target_velocity,
        target_velocity=target_velocity,
        **kwargs,
    )["H"]
    h_minus = corecbf_terms(
        ship_state=ship - eps * ship_dot,
        target_position=target_position - eps * target_velocity,
        target_velocity=target_velocity,
        **kwargs,
    )["H"]

    numerical = (h_plus - h_minus) / (2.0 * eps)
    analytic = terms["LfH"] + terms["LgH"] @ control
    assert analytic == pytest.approx(numerical, rel=2e-6, abs=2e-6)

    with pytest.raises(ValueError, match="turn_direction"):
        corecbf_terms(
            ship_state=ship,
            target_position=target_position,
            target_velocity=target_velocity,
            **{**kwargs, "turn_direction": 1.5},
        )


def test_multi_obstacle_qp_uses_one_shared_slack_and_exact_propeller_bounds():
    solved = solve_corecbf_qp(
        nominal_control=np.zeros(2),
        h=np.zeros(2),
        lf_h=-np.ones(2),
        lg_h=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        cbf_gain=1.0,
        y_p=0.5,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        weights=np.ones(2),
        slack_weight=100.0,
    )

    assert solved["control"] == pytest.approx([0.0, 0.0], abs=2e-5)
    assert solved["slack"] == pytest.approx(1.0, abs=2e-5)
    assert solved["residuals"] == pytest.approx([-1.0, -1.0], abs=2e-5)
    assert solved["relaxed_residuals"] == pytest.approx([0.0, 0.0], abs=2e-5)
    assert np.all(solved["thrusts"] >= -2.0)
    assert np.all(solved["thrusts"] <= 3.0)


def test_multi_obstacle_qp_uses_rowwise_slack_when_enabled():
    solved = solve_corecbf_qp(
        nominal_control=np.zeros(2),
        h=np.zeros(2),
        lf_h=np.array([-1.0, 0.0]),
        lg_h=np.zeros((2, 2)),
        cbf_gain=1.0,
        y_p=0.5,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        weights=np.ones(2),
        slack_weight=100.0,
        separable_slack=True,
    )

    assert solved["slacks"] == pytest.approx([1.0, 0.0], abs=2e-5)
    assert solved["slack"] == pytest.approx(1.0, abs=2e-5)
    assert solved["slack_sum"] == pytest.approx(1.0, abs=2e-5)
    assert solved["slack_positive_count"] == 1
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_separable_slack_fallback_stays_policy_close_at_osqp_iteration_limit():
    solved = solve_corecbf_qp(
        nominal_control=np.zeros(2),
        h=np.zeros(1),
        lf_h=np.array([-4.0]),
        lg_h=np.array([[1.0, 0.0]]),
        cbf_gain=1.0,
        y_p=0.5,
        thrust_lower=-3.0,
        thrust_upper=3.0,
        weights=np.ones(2),
        slack_weight=100.0,
        separable_slack=True,
        max_iter=1,
        polishing=False,
    )

    # The minimum-slack LP has many safe optima here.  If OSQP is interrupted,
    # the fallback must retain its stage-2 policy-closeness objective rather
    # than return an arbitrary LP extreme point such as [3, 3].
    assert solved["thrusts"] == pytest.approx([2.0, 2.0], abs=2e-5)
    assert solved["slack_sum"] == pytest.approx(0.0, abs=2e-5)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)
    assert "stable fallback" in solved["status"]


def test_separable_slack_requires_boolean_flag():
    with pytest.raises(ValueError, match="separable_slack must be boolean"):
        solve_corecbf_qp(
            nominal_control=np.zeros(2),
            h=np.zeros(1),
            lf_h=np.zeros(1),
            lg_h=np.zeros((1, 2)),
            cbf_gain=1.0,
            y_p=0.5,
            thrust_lower=-2.0,
            thrust_upper=3.0,
            weights=np.ones(2),
            slack_weight=100.0,
            separable_slack=1,
        )


def test_qp_uses_zero_slack_whenever_the_hard_constraint_is_feasible():
    solved = solve_corecbf_qp(
        nominal_control=np.zeros(2),
        h=np.zeros(1),
        lf_h=-np.ones(1),
        lg_h=np.array([[1.0, 0.0]]),
        cbf_gain=1.0,
        y_p=0.5,
        thrust_lower=-2.0,
        thrust_upper=3.0,
        weights=np.ones(2),
        slack_weight=1.0,
    )

    assert solved["slack"] == pytest.approx(0.0, abs=2e-5)
    assert solved["control"] == pytest.approx([1.0, 0.0], abs=2e-5)
    assert solved["residuals"] == pytest.approx([0.0], abs=2e-5)


def test_qp_repairs_solver_tolerance_at_exact_thrust_bound():
    solved = solve_corecbf_qp(
        nominal_control=np.array([127.50016568678207, 44.18621455375738]),
        h=np.array([66.02218975355491, 47.06981194858197]),
        lf_h=np.array([-43.823343463336684, -109.02009495295948]),
        lg_h=np.array(
            [
                [-0.06233194136877157, -0.6343403354255612],
                [0.658239115810144, -0.515681654719372],
            ]
        ),
        cbf_gain=0.25,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=10000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_slack_scaling_avoids_max_iterations_for_nav_case():
    solved = solve_corecbf_qp(
        nominal_control=np.array([-70.76157899911887, 24.74849629553772]),
        h=np.array([50.6506830155086, 46.203805922045134]),
        lf_h=np.array([-119.6955706459028, -111.34409730936461]),
        lg_h=np.array(
            [
                [0.4370604848040852, -0.5431938904238275],
                [0.5187077828873243, -0.22553793969234096],
            ]
        ),
        cbf_gain=0.25,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=10000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_multi_row_nav_case_converges_with_shared_slack():
    solved = solve_corecbf_qp(
        nominal_control=np.array([1.7809295891805164, 53.402787187915976]),
        h=np.array([1.3507739351645238, 54.524014336268074, 50.69528656215245]),
        lf_h=np.array([-17.968898252596638, -102.71926254341214, -126.0074374686147]),
        lg_h=np.array(
            [
                [-0.011388283267195543, -0.03686969404343486],
                [0.44933536538920155, -0.1056378576125873],
                [0.24308177865534814, -0.5994663674725127],
            ]
        ),
        cbf_gain=0.25,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=10000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_qp_repairs_small_residual_error_with_large_slack_weight():
    solved = solve_corecbf_qp(
        nominal_control=np.array([16.938059031423293, 59.38985331760188]),
        h=np.array([1.9254342718396154]),
        lf_h=np.array([-0.01325936463580435]),
        lg_h=np.array([[-0.09352365313007022, -0.05355469215359962]]),
        cbf_gain=1.0,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=1000000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)
    assert solved["status"].endswith("projected")


def test_qp_projects_solved_inaccurate_slack_to_exact_feasibility():
    solved = solve_corecbf_qp(
        nominal_control=np.array([-72.68128683596973, -18.954152809782805]),
        h=np.array([0.05222149651865357]),
        lf_h=np.array([-0.194083745225389]),
        lg_h=np.array([[-0.0013135010584983964, -8.344003343790809e-06]]),
        cbf_gain=1.0,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=1000000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)
    assert solved["status"].endswith("projected")


def test_large_slack_weight_converges_for_small_input_coefficients():
    solved = solve_corecbf_qp(
        nominal_control=np.array([-54.44864549917294, -8.082041145897982]),
        h=np.array([1.346112360701719]),
        lf_h=np.array([-2.3229066167837797]),
        lg_h=np.array([[-0.013842347764775203, -0.006018094319434711]]),
        cbf_gain=1.0,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=1000000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_feasible_warm_start_handles_nearly_degenerate_input_row():
    solved = solve_corecbf_qp(
        nominal_control=np.array([-66.20320445006925, 5.817036609251276]),
        h=np.array([-0.042107196889448285]),
        lf_h=np.array([-0.0863065102298634]),
        lg_h=np.array([[-0.0002968497728118632, -3.4172693151919657e-06]]),
        cbf_gain=1.0,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=1000000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_extreme_multirow_case_returns_a_feasible_result():
    solved = solve_corecbf_qp(
        nominal_control=np.array([-66.95659996129794, 26.251463015476986]),
        h=np.array(
            [
                0.08933918526630637,
                4.350769092906035,
                28.734803447062625,
                1.9497242550409624,
                22.335344394252658,
            ]
        ),
        lf_h=np.array(
            [
                -0.47895116691111217,
                -6.644604887257445,
                -5.498422871802969,
                -8.129940769481568,
                18.10016398710574,
            ]
        ),
        lg_h=np.array(
            [
                [0.0029747152098412006, -1.301718840968912e-05],
                [0.058694189096250224, -0.022289744319420733],
                [-0.019673714487774052, -0.22703806350608202],
                [0.06065421076140809, -0.04857068235872063],
                [-0.25093935460850203, -0.30442414221890657],
            ]
        ),
        cbf_gain=1.0,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=1000000.0,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


@pytest.mark.parametrize(
    ("nominal_control", "h", "lf_h", "lg_h", "cbf_gain", "slack_weight"),
    [
        (
            [54.96945837914802, 72.40447570959975],
            [51.58828199818752, 32.25852631542633, 16.776056068672787, 6.0087634269489385],
            [-51.18709297436596, -37.2079838441253, -16.67466449482337, -9.324136453544638],
            [
                [0.05466842374910735, -0.6411793905896521],
                [0.048836465621334924, -0.4045872314691883],
                [-0.08275753856247989, -0.15940771455518418],
                [-0.0210521960319206, -0.06200280603085296],
            ],
            0.25,
            10000.0,
        ),
        (
            [-58.8096907891193, -15.567806129786051],
            [0.9725283589050935],
            [-1.431404633486533],
            [[-0.006844096555276636, -0.002798287709685588]],
            1.0,
            1000000.0,
        ),
    ],
)
def test_default_iteration_budget_covers_extreme_nav_cases(
    nominal_control, h, lf_h, lg_h, cbf_gain, slack_weight
):
    solved = solve_corecbf_qp(
        nominal_control=np.asarray(nominal_control),
        h=np.asarray(h),
        lf_h=np.asarray(lf_h),
        lg_h=np.asarray(lg_h),
        cbf_gain=cbf_gain,
        y_p=0.395,
        thrust_lower=-66.70800000024009,
        thrust_upper=119.68200000004595,
        weights=np.ones(2),
        slack_weight=slack_weight,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )

    assert np.all(solved["thrusts"] >= -66.70800000024009)
    assert np.all(solved["thrusts"] <= 119.68200000004595)
    assert np.all(solved["relaxed_residuals"] >= -1e-10)


def test_corecbf_terms_remain_finite_inside_design_distance_for_engineering_recovery():
    terms = corecbf_terms(
        ship_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        target_position=np.array([1.8, 0.0]),
        target_velocity=np.zeros(2),
        safety_distance=2.0,
        surge_accel=1.0,
        turn_accel=1.0,
        turn_direction=-1,
        usv_params=PARAMS,
    )

    assert terms["distance"] == pytest.approx(1.8)
    assert np.all(np.isfinite([terms["H"], terms["LfH"], *terms["LgH"]]))


def test_otter_routes_all_gated_obstacles_through_one_corecbf_qp():
    env = USVEnvMinimal(
        grid_map=np.zeros((16, 16), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        corecbf_surge_accel=0.8,
        corecbf_turn_accel=1.2,
        corecbf_turn_direction=-1,
        corecbf_gain=0.7,
        corecbf_safety_distance=2.0,
    )
    try:
        assert env.corecbf_turn_direction == -1
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {"dx": 5.0, "dy": 0.0, "vx": 0.0, "vy": 0.0, "radius": 1.0},
            1: {"dx": 6.0, "dy": 0.0, "vx": 0.0, "vy": 0.0, "radius": 1.0},
        }
        env.civo_cones = {0: {}, 1: {}}
        filtered = env._shield_action_civo(np.zeros(2, dtype=np.float32))
        assert filtered.shape == (2,)
        assert np.all(np.isfinite(filtered))
        assert env._last_civo_constraint_count == 2
        assert env._last_civo_obstacle_id in {0, 1}
        assert env._last_civo_slack >= 0.0

        env.obstacle_estimates.pop(1)
        env.civo_cones = {0: {}}
        filtered = env._shield_action_civo(np.zeros(2, dtype=np.float32))
        assert filtered == pytest.approx([0.0, 0.0])
        assert env._last_civo_qp_status == "nominal feasible"
        assert env._last_civo_constraint_count == 1
        assert env._last_civo_obstacle_id == 0
        assert env._last_civo_residual >= 0.0
    finally:
        env.close()


def test_otter_all_obstacles_mode_bypasses_distance_tcpa_gate():
    env = USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        civo_shield_enabled=True,
        civo_shield_gate_mode="all_obstacles",
        corecbf_safety_distance=2.0,
    )
    try:
        env.ship_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        env.obstacle_estimates = {
            0: {"dx": 5.0, "dy": 0.0, "vx": 0.0, "vy": 0.0, "radius": 1.0},
            1: {"dx": 20.0, "dy": 0.0, "vx": 0.0, "vy": 0.0, "radius": 1.0},
        }
        env.civo_cones = {0: {}, 1: {}}

        env._shield_action_civo(np.zeros(2, dtype=np.float32))

        assert env._last_civo_constraint_count == 2
    finally:
        env.close()


def test_env_config_maps_only_the_new_corecbf_parameters():
    kwargs = build_env_kwargs(
        {
            "civo": {
                "shield_enabled": True,
                "surge_accel": 0.8,
                "turn_accel": 1.2,
                "turn_direction": -1,
                "cbf_gain": 0.7,
                "safety_distance": 2.5,
                "colregs_reference_scale": 0.25,
                "qp_tau_u_weight": 2.0,
                "qp_tau_r_weight": 3.0,
                "shared_slack_weight": 10000.0,
                "separable_slack_enabled": True,
            }
        },
        Path("scenarios"),
    )

    assert kwargs["corecbf_surge_accel"] == 0.8
    assert kwargs["corecbf_turn_accel"] == 1.2
    assert kwargs["corecbf_turn_direction"] == -1
    assert kwargs["corecbf_gain"] == 0.7
    assert kwargs["corecbf_safety_distance"] == 2.5
    assert kwargs["corecbf_colregs_reference_scale"] == 0.25
    assert kwargs["corecbf_slack_weight"] == 10000.0
    assert kwargs["corecbf_separable_slack_enabled"] is True
    assert build_env_kwargs({}, Path("scenarios"))[
        "corecbf_colregs_reference_scale"
    ] == 0.1875
    assert "civo_hocbf_qp_enabled" not in kwargs

    with pytest.raises(ValueError, match="retired CoReCBF config"):
        build_env_kwargs({"civo": {"hocbf_gain": 1.0}}, Path("scenarios"))


def test_env_config_maps_shield_gate_mode_with_distance_tcpa_default():
    default = build_env_kwargs({}, Path("scenarios"))
    configured = build_env_kwargs(
        {"civo": {"shield_gate_mode": "all_obstacles"}},
        Path("scenarios"),
    )

    assert default["civo_shield_gate_mode"] == "distance_tcpa"
    assert configured["civo_shield_gate_mode"] == "all_obstacles"
