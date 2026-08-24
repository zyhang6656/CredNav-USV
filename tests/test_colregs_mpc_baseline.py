import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_colregs_mpc_baseline import aggregate, resolve_eval_sources
from simple_boat.envs.colregs_mpc_baseline import COLREGSMPCBaseline
from simple_boat.envs.usv_env_minimal import USVEnvMinimal


def _make_static_env() -> USVEnvMinimal:
    return USVEnvMinimal(
        grid_map=np.zeros((32, 32), dtype=np.uint8),
        dynamic_obstacles=False,
        fixed_initial_position=np.array([0.0, 0.0], dtype=np.float32),
        fixed_initial_psi=0.0,
        fixed_goal=np.array([31.0, 0.0], dtype=np.float32),
    )


def test_colregs_mpc_activates_hard_constraint_for_give_way_risk():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([4.0, 16.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        env.goal = np.array([28.0, 16.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 8.0, "dy": -2.0, "vx": -0.3, "vy": 0.1, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert action.shape == (2,)
    assert info["colregs_role"] == "GW"
    assert info["colregs_mpc_active_constraints"] > 0


def test_colregs_mpc_uses_straight_reference():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        env.goal = np.array([31.0, 0.0], dtype=float)

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert action.shape == (2,)
    assert info["colregs_mpc_reference"] == "straight"


def test_colregs_mpc_role_fsm_holds_until_encounter_exit():
    env = _make_static_env()
    controller = COLREGSMPCBaseline(encounter_radius=12.0)
    try:
        env.reset(seed=0)
        env.ship_state = np.array([4.0, 16.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        env.goal = np.array([28.0, 16.0], dtype=float)
        env.obstacle_estimates = {
            7: {"dx": 8.0, "dy": -2.0, "vx": -0.3, "vy": 0.1, "radius": 1.0},
        }
        _action, info = controller.predict(env)
        assert info["colregs_role"] == "GW"

        env.obstacle_estimates = {
            7: {"dx": 8.0, "dy": -2.0, "vx": 0.8, "vy": -0.2, "radius": 1.0},
        }
        _action, info = controller.predict(env)
        assert info["colregs_role"] == "GW"

        env.obstacle_estimates = {
            7: {"dx": 14.0, "dy": -2.0, "vx": 0.8, "vy": -0.2, "radius": 1.0},
        }
        _action, info = controller.predict(env)
        assert info["colregs_role"] == "NONE"
    finally:
        env.close()


def test_colregs_mpc_solves_close_give_way_risk_with_constraints():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([8.85, 8.80, 0.81, 0.35, 0.0, 0.0], dtype=float)
        env.goal = np.array([27.0, 27.0], dtype=float)
        env.obstacle_estimates = {
            2: {"dx": 7.57, "dy": -0.2, "vx": -1.6, "vy": 0.1, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "GW"
    assert info["colregs_mpc_active_constraints"] > 0
    assert np.all(np.isfinite(action))


def test_colregs_mpc_keeps_hard_constraint_until_encounter_exit():
    env = _make_static_env()
    controller = COLREGSMPCBaseline()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([16.0, 16.6, 1.76, 0.2, 0.0, 0.0], dtype=float)
        env.goal = np.array([27.0, 27.0], dtype=float)
        env.obstacle_estimates = {
            2: {"dx": -3.0, "dy": 7.0, "vx": -0.5, "vy": -1.0, "radius": 1.0},
        }
        controller._roles[2] = "GW"

        action, info = controller.predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "GW"
    assert info["colregs_mpc_active_constraints"] > 0
    assert action.shape == (2,)


def test_colregs_mpc_uses_casadi_ipopt_and_not_legacy_solvers():
    source = pathlib.Path(colregs_mpc_baseline_path()).read_text(encoding="utf-8")

    assert "scipy.optimize" not in source
    assert "SLSQP" not in source
    assert "osqp" not in source.lower()
    assert "_surge_candidates" not in source
    assert "_yaw_candidates" not in source
    assert "control_knots" not in source


def colregs_mpc_baseline_path() -> pathlib.Path:
    return REPO_ROOT / "simple_boat" / "envs" / "colregs_mpc_baseline.py"


def test_colregs_mpc_reports_casadi_ipopt_solver_with_hard_constraints():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([4.0, 16.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        env.goal = np.array([28.0, 16.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 8.0, "dy": -2.0, "vx": -0.3, "vy": 0.1, "radius": 1.0},
        }

        _action, info = COLREGSMPCBaseline(horizon_steps=6, optimizer_maxiter=20).predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "GW"
    assert info["colregs_mpc_solver"] == "casadi_ipopt"
    assert info["colregs_mpc_solver_status"] != "not_solved"
    assert info["colregs_mpc_active_constraints"] > 0


def test_colregs_mpc_rejects_solver_plan_that_violates_hard_constraints(monkeypatch):
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=2)

    class FakeSolver:
        def __call__(self, **kwargs):
            return {"x": kwargs["x0"], "f": 1.0}

        def stats(self):
            return {"success": True}

    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        previous = np.tile(np.array([0.2, 0.0], dtype=float), (2, 1))
        controller._last_actions = previous.copy()
        monkeypatch.setattr(controller, "_casadi_solver", lambda n_rows: {
            "solver": FakeSolver(),
            "lbx": np.array([], dtype=float),
            "ubx": np.array([], dtype=float),
            "lbg": np.array([], dtype=float),
            "ubg": np.array([], dtype=float),
        })
        monkeypatch.setattr(controller, "_colregs_constraint_rows", lambda env, items: [{
            "k": 0,
            "normal": np.array([1.0, 0.0], dtype=float),
            "point": np.array([0.0, 0.0], dtype=float),
            "rho": 1.0,
        }])
        monkeypatch.setattr(controller, "_constraint_values_for_actions", lambda env, actions, rows: np.array([-0.1], dtype=float))

        action, score, failed = controller._optimize_action_sequence(env, [{
            "role": "GW",
            "dx": 1.0,
            "dy": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "tcpa": 1.0,
            "dcpa": 0.0,
            "dist": 1.0,
        }], "GW")
    finally:
        env.close()

    assert failed is True
    assert np.isinf(score)
    assert np.allclose(action, previous[0])


def test_colregs_mpc_solver_failure_fallback_shifts_previous_plan(monkeypatch):
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=3)

    try:
        env.reset(seed=0)
        previous = np.array([[0.1, 0.0], [0.2, 0.1], [0.3, 0.2]], dtype=float)
        controller._last_actions = previous.copy()
        monkeypatch.setattr(controller, "_casadi_solver", lambda n_rows: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(controller, "_colregs_constraint_rows", lambda env, items: [])

        first, _score, failed = controller._optimize_action_sequence(env, [], "NONE")
        second, _score, failed_again = controller._optimize_action_sequence(env, [], "NONE")
    finally:
        env.close()

    assert failed is True
    assert failed_again is True
    assert np.allclose(first, previous[0])
    assert np.allclose(second, previous[1])


def test_colregs_mpc_reports_nonconverged_but_feasible_ipopt_plan(monkeypatch):
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=2)

    class FakeSolver:
        def __call__(self, **kwargs):
            return {"x": kwargs["x0"], "f": 1.0}

        def stats(self):
            return {"success": False, "return_status": "Maximum_Iterations_Exceeded"}

    try:
        env.reset(seed=0)
        monkeypatch.setattr(controller, "_casadi_solver", lambda n_rows: {
            "solver": FakeSolver(),
            "lbx": np.array([], dtype=float),
            "ubx": np.array([], dtype=float),
            "lbg": np.array([], dtype=float),
            "ubg": np.array([], dtype=float),
        })
        monkeypatch.setattr(controller, "_colregs_constraint_rows", lambda env, items: [])

        action, score, failed = controller._optimize_action_sequence(env, [], "NONE")
    finally:
        env.close()

    assert np.all(np.isfinite(action))
    assert score == 1.0
    assert failed is True
    assert controller._last_solver_status == "Maximum_Iterations_Exceeded"


def test_colregs_mpc_does_not_double_count_physical_safety_radius():
    controller = COLREGSMPCBaseline(safety_distance=2.0, ov_length=2.0, ov_width=1.08)

    vertices = controller._rectangle_vertices(
        np.array([0.0, 0.0], dtype=float),
        0.0,
    )

    assert any(np.allclose(v, np.array([1.0, 0.54], dtype=float)) for v in vertices)
    assert any(np.allclose(v, np.array([1.0, -0.54], dtype=float)) for v in vertices)


def test_colregs_mpc_uses_paper_asymmetric_geometry_margins():
    controller = COLREGSMPCBaseline(ov_length=2.0, ov_width=1.08)

    vertices = controller._rectangle_vertices(
        np.array([0.0, 0.0], dtype=float),
        0.0,
        margins=(2.0, 1.0, 1.08, 1.08),
    )

    assert any(np.allclose(v, np.array([3.0, 1.62], dtype=float)) for v in vertices)
    assert any(np.allclose(v, np.array([3.0, -1.62], dtype=float)) for v in vertices)
    assert any(np.allclose(v, np.array([-2.0, -1.62], dtype=float)) for v in vertices)
    assert any(np.allclose(v, np.array([-2.0, 1.62], dtype=float)) for v in vertices)


def test_colregs_mpc_scales_paper_geometry_margins_for_environment():
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=1, geometry_margin_scale=0.5)
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        controller._last_pred_positions = np.array([[10.0, 0.0]], dtype=float)
        rows = controller._colregs_constraint_rows(env, [{
            "id": 0,
            "role": "GW",
            "dx": 5.0,
            "dy": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "tcpa": 1.0,
            "dcpa": 0.0,
            "dist": 5.0,
            "radius": 1.0,
        }])
    finally:
        env.close()

    assert abs(float(rows[0]["point"][1])) >= 1.08
    assert abs(float(rows[0]["point"][1])) < 1.62


def test_colregs_mpc_uses_rectangle_vertex_for_paper_halfspace():
    controller = COLREGSMPCBaseline(horizon_steps=1, ov_length=2.0, ov_width=1.08)

    row = controller._constraint_row(
        0,
        np.array([10.0, 0.1], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        0.0,
        "GW",
        1.0,
    )

    vertices = controller._rectangle_vertices(np.array([0.0, 0.0], dtype=float), 0.0)
    assert any(np.allclose(row["point"], vertex) for vertex in vertices)


def test_colregs_mpc_halfspace_always_supports_inflated_rectangle():
    controller = COLREGSMPCBaseline(horizon_steps=1, ov_length=2.0, ov_width=1.08)
    center = np.array([0.0, 0.0], dtype=float)
    margins = (2.0, 1.0, 1.08, 1.08)

    row = controller._constraint_row(
        0,
        np.array([0.0, 0.0], dtype=float),
        center,
        0.0,
        "GW",
        1.0,
        margins=margins,
    )

    vertices = controller._rectangle_vertices(center, 0.0, margins=margins)
    support_errors = [
        float(row["normal"] @ (vertex - row["point"]))
        for vertex in vertices
    ]
    assert max(support_errors) <= 1e-9


def test_colregs_mpc_rotates_give_way_constraint_to_starboard_in_env_frame():
    controller = COLREGSMPCBaseline(horizon_steps=1, ov_length=2.0, ov_width=1.08)

    row = controller._constraint_row(
        0,
        np.array([10.0, 0.1], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        0.0,
        "GW",
        1.0,
    )

    assert row["normal"][1] > 0.0


def test_colregs_mpc_give_way_commands_starboard_yaw_in_env_frame():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([4.0, 16.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        env.goal = np.array([28.0, 16.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 8.0, "dy": -2.0, "vx": -0.3, "vy": 0.1, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline(horizon_steps=20, optimizer_maxiter=30).predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "GW"
    assert action[1] < 0.0


def test_colregs_mpc_uses_asv_radius_for_paper_halfspace():
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=1, safety_distance=0.5)
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        controller._last_pred_positions = np.array([[10.0, 0.0]], dtype=float)
        rows = controller._colregs_constraint_rows(env, [{
            "id": 0,
            "role": "GW",
            "dx": 5.0,
            "dy": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "tcpa": 1.0,
            "dcpa": 0.0,
            "dist": 5.0,
            "radius": 1.0,
        }])
    finally:
        env.close()

    assert np.isclose(rows[0]["rho"], env.R_usv)


def test_colregs_mpc_augments_ov_rectangle_to_cover_env_obstacle_radius():
    env = _make_static_env()
    controller = COLREGSMPCBaseline(horizon_steps=1, ov_length=2.0, ov_width=1.08)
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        controller._last_pred_positions = np.array([[10.0, 0.0]], dtype=float)
        rows = controller._colregs_constraint_rows(env, [{
            "id": 0,
            "role": "GW",
            "dx": 5.0,
            "dy": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "tcpa": 1.0,
            "dcpa": 0.0,
            "dist": 5.0,
            "radius": 1.0,
        }])
    finally:
        env.close()

    assert abs(float(rows[0]["point"][1])) >= 1.62


def test_colregs_mpc_default_geometry_and_fsm_thresholds_match_paper_table_ii():
    controller = COLREGSMPCBaseline()

    assert controller.horizon_steps == 41
    assert math.isclose(controller.prediction_dt, 0.25)
    assert math.isclose(controller.control_period, 0.25)
    assert controller.encounter_radius == 21.0
    assert controller.emergency_radius == 10.0
    assert math.isclose(math.radians(controller.head_on_course_deg), 0.25)
    assert math.isclose(controller.colregs_alpha_gw, 0.97)


def test_colregs_mpc_objective_weights_keep_paper_table_i_ratios():
    controller = COLREGSMPCBaseline()

    assert math.isclose(controller.q_lag, 0.001)
    assert math.isclose(controller.q_contour, 0.1)
    assert math.isclose(controller.q_speed_em, 0.01)
    assert math.isclose(controller.q_speed, 1.0)
    assert math.isclose(controller.q_lateral_velocity, 0.25)
    assert math.isclose(controller.q_control_surge, 0.0001)
    assert math.isclose(controller.q_control_yaw, 0.003)


def test_colregs_mpc_paper_risk_gate_has_no_tcpa_upper_limit():
    env = _make_static_env()
    controller = COLREGSMPCBaseline()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        role = controller._entry_role(
            env,
            {"dx": 20.0, "dy": -0.2, "vx": -0.8, "vy": 0.0, "radius": 1.0},
            tcpa=100.0,
            dcpa=1.0,
            dist=20.0,
        )
    finally:
        env.close()

    assert role == "GW"


def test_colregs_mpc_risk_limit_adds_both_radii_and_safety_distance():
    env = _make_static_env()
    controller = COLREGSMPCBaseline(safety_distance=0.5)
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        role = controller._entry_role(
            env,
            {"dx": 5.0, "dy": -0.2, "vx": -0.8, "vy": 0.0, "radius": 1.0},
            tcpa=2.0,
            dcpa=2.4,
            dist=5.0,
        )
    finally:
        env.close()

    assert role == "GW"


def test_colregs_mpc_entry_role_uses_paper_eq23_not_bearing_alone():
    env = _make_static_env()
    controller = COLREGSMPCBaseline()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        role = controller._entry_role(
            env,
            {"dx": 5.0, "dy": -2.0, "vx": 0.0, "vy": -1.0, "radius": 1.0},
            tcpa=2.0,
            dcpa=1.0,
            dist=5.0,
        )
    finally:
        env.close()

    assert role == "SO"


def test_colregs_mpc_ignores_distant_non_collision_bearing():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([5.0, 10.0, 0.7, 0.0, 0.0, 0.0], dtype=float)
        env.goal = np.array([27.0, 27.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 12.0, "dy": -16.0, "vx": 0.7, "vy": 1.1, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "NONE"
    assert action.shape == (2,)


def test_colregs_mpc_ignores_receding_close_obstacle_outside_safety_margin():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([12.0, 14.0, 5.6, 0.6, 0.0, 0.0], dtype=float)
        env.goal = np.array([27.0, 27.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 4.6, "dy": -1.4, "vx": 0.9, "vy": -0.7, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "NONE"


def test_colregs_mpc_turns_toward_path_when_heading_away():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([11.84, 4.46, 4.45, 0.23, 0.0, 0.0], dtype=float)
        env.goal = np.array([27.0, 27.0], dtype=float)
        env.obstacle_estimates = {}

        action, info = COLREGSMPCBaseline().predict(env)
    finally:
        env.close()

    assert info["colregs_role"] == "NONE"
    assert info["colregs_mpc_reference"] == "straight"


def test_colregs_mpc_initial_guess_keeps_heading_continuous_across_zero():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([10.0, 10.0, 0.02, 0.8, 0.0, -0.4], dtype=float)

        controller = COLREGSMPCBaseline(horizon_steps=8)
        controller._last_actions = np.tile(np.array([0.55, -0.45], dtype=float), (8, 1))
        X, _U, _S, _V = controller._unpack_decision(controller._initial_decision_guess(env))
    finally:
        env.close()

    assert float(np.max(np.abs(np.diff(X[2])))) < math.pi


def test_colregs_mpc_first_give_way_guess_starts_with_starboard_turn():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        controller = COLREGSMPCBaseline(horizon_steps=8)

        _X, U, _S, _V = controller._unpack_decision(controller._initial_decision_guess(env, role="GW"))
    finally:
        env.close()

    assert np.allclose(U[0], controller.cruise_action)
    assert np.all(U[1] < 0.0)


def test_colregs_mpc_replaces_infeasible_straight_warm_start_for_give_way(monkeypatch):
    env = _make_static_env()
    try:
        env.reset(seed=0)
        controller = COLREGSMPCBaseline(horizon_steps=8)
        controller._last_actions = np.tile(np.array([0.55, 0.0], dtype=float), (8, 1))
        monkeypatch.setattr(
            controller,
            "_constraint_values_for_actions",
            lambda _env, actions, _rows: np.array([float(np.min(-actions[:, 1])) - 0.1]),
        )

        _X, U, _S, _V = controller._unpack_decision(
            controller._initial_decision_guess(env, role="GW", rows=[{"constraint": "GW"}])
        )
    finally:
        env.close()

    assert np.all(U[1] < 0.0)


def test_colregs_mpc_holds_action_between_control_periods(monkeypatch):
    env = _make_static_env()
    controller = COLREGSMPCBaseline(control_period=0.25)
    calls = {"count": 0}

    def fake_optimize(env, role_items, role):
        calls["count"] += 1
        return np.array([0.2 * calls["count"], -0.1], dtype=float), 1.0, False

    try:
        env.reset(seed=0)
        monkeypatch.setattr(controller, "_optimize_action_sequence", fake_optimize)

        env.dyn_step = 0
        first, first_info = controller.predict(env)
        env.dyn_step = 1
        second, second_info = controller.predict(env)
        env.dyn_step = 2
        third, third_info = controller.predict(env)
        env.dyn_step = 3
        fourth, fourth_info = controller.predict(env)
    finally:
        env.close()

    assert calls["count"] == 2
    assert np.allclose(second, first)
    assert np.allclose(third, first)
    assert not np.allclose(fourth, first)
    assert first_info["colregs_mpc_held"] is False
    assert second_info["colregs_mpc_held"] is True
    assert third_info["colregs_mpc_held"] is True
    assert fourth_info["colregs_mpc_held"] is False


def test_colregs_mpc_keeps_predicted_positions_inside_env_bounds():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([16.0, 30.5, 0.5 * math.pi, 1.0, 0.0, 0.0], dtype=float)
        env.goal = np.array([16.0, 31.0], dtype=float)

        controller = COLREGSMPCBaseline(horizon_steps=4, optimizer_maxiter=20)
        action, info = controller.predict(env)
        predicted = controller._rollout_states(env, np.tile(action, (4, 1)))
    finally:
        env.close()

    assert info["colregs_mpc_solver_failed"] is False
    assert float(np.max(predicted[:, 1])) <= 31.0 + 1e-3


def test_colregs_mpc_solver_includes_position_bound_constraints():
    controller = COLREGSMPCBaseline(horizon_steps=2)
    data = controller._casadi_solver(0)

    assert len(data["lbg"]) == 7 + 2 * 12 + 1


def test_colregs_mpc_path_parameter_is_advanced_by_surge_speed_not_free_virtual_speed():
    controller = COLREGSMPCBaseline(horizon_steps=3)
    data = controller._casadi_solver(0)

    assert len(data["lbx"]) == 6 * 4 + 2 * 3 + 4


def test_colregs_mpc_path_parameter_can_progress_beyond_goal():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.initial_position = np.array([0.0, 16.0], dtype=float)
        env.goal = np.array([31.0, 16.0], dtype=float)
        env.ship_state = np.array([29.5, 16.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        controller = COLREGSMPCBaseline(horizon_steps=8, prediction_dt=0.25)

        _X, _U, S, _V = controller._unpack_decision(controller._initial_decision_guess(env))
    finally:
        env.close()

    assert S[-1] > 31.0


def test_colregs_mpc_emergency_uses_hard_constraints_not_forced_speed_cap():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        env.ship_state = np.array([4.0, 16.0, 0.0, 0.8, 0.0, 0.0], dtype=float)
        env.goal = np.array([28.0, 16.0], dtype=float)
        env.obstacle_estimates = {
            0: {"dx": 3.0, "dy": 2.0, "vx": -0.4, "vy": -0.2, "radius": 1.0},
        }

        action, info = COLREGSMPCBaseline(horizon_steps=6, optimizer_maxiter=20).predict(env)
    finally:
        env.close()

    assert action.shape == (2,)
    assert info["colregs_role"] == "EM"
    assert info["colregs_mpc_active_constraints"] > 0


def test_colregs_mpc_one_step_prediction_matches_otter_model():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        action = np.array([0.5, -0.25], dtype=np.float32)

        expected = env._predict_next_state_for_action(action)
        actual = COLREGSMPCBaseline(prediction_dt=env.dt)._next_state_for_action(env, env.ship_state.copy(), action)

        assert np.allclose(actual, expected)
    finally:
        env.close()


def test_colregs_mpc_casadi_step_matches_otter_model():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        state = np.array([4.0, 16.0, 0.2, 0.5, 0.03, -0.02], dtype=float)
        action = np.array([0.4, -0.2], dtype=float)

        controller = COLREGSMPCBaseline(prediction_dt=env.dt)
        expected = controller._next_state_for_action(env, state, action)
        actual = controller._casadi_step_numeric(env, state, action)

        assert np.allclose(actual, expected, atol=1e-9)
    finally:
        env.close()


def test_colregs_mpc_substeps_match_two_environment_steps():
    env = _make_static_env()
    try:
        env.reset(seed=0)
        state = np.array([4.0, 16.0, 0.2, 0.5, 0.03, -0.02], dtype=float)
        action = np.array([0.4, -0.2], dtype=float)
        controller = COLREGSMPCBaseline(prediction_dt=0.2, integration_substeps=2)

        expected = state.copy()
        for _ in range(2):
            expected = COLREGSMPCBaseline(prediction_dt=env.dt)._next_state_for_action(env, expected, action)
        numeric = controller._next_state_for_action(env, state, action)
        symbolic = controller._casadi_step_numeric(env, state, action)
    finally:
        env.close()

    assert np.allclose(numeric, expected, atol=1e-9)
    assert np.allclose(symbolic, expected, atol=1e-9)


def test_resolve_eval_sources_defaults_to_obs3_through_obs6():
    sources = resolve_eval_sources({"data": {}})

    assert [s["label"] for s in sources] == ["obs3", "obs4", "obs5", "obs6"]
    assert [s["scenario_dir"].as_posix() for s in sources] == [
        "simple_boat/assets/eval3_new_map",
        "simple_boat/assets/eval4_new_map",
        "simple_boat/assets/eval5_new_map",
        "simple_boat/assets/eval6_new_map",
    ]


def test_colregs_mpc_aggregate_reports_solver_failure_rate():
    row = {
        "raw_success": 0,
        "strict_success": 0,
        "collision": 1,
        "timeout": 0,
        "unsafe_near_miss": 0,
        "return": -1.0,
        "path_length": 2.0,
        "min_actual_distance": 1.5,
        "min_dcpa": 0.2,
        "colregs_compliance": 0.8,
        "colregs_mpc_ms_mean": 10.0,
        "colregs_mpc_ms_p95": 20.0,
        "colregs_mpc_solver_failed_steps": 3,
        "colregs_mpc_solver_failed_rate": 0.25,
        "steps": 12,
    }

    summary = aggregate([row])

    assert summary["colregs_mpc_solver_failed_steps_mean"] == 3.0
    assert summary["colregs_mpc_solver_failed_rate_mean"] == 0.25
