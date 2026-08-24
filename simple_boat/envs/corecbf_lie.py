import math

import numpy as np


class CredibleGeometryDomainError(ValueError):
    """Valid inputs lie outside the credible-geometry theoretical domain."""


def _finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-{shape} array")
    return array


def _finite_cbf_constraints(h, lf_h, lg_h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        h = np.asarray(h, dtype=float).reshape(-1)
        lf_h = np.asarray(lf_h, dtype=float).reshape(-1)
        lg_h = np.asarray(lg_h, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CBF rows must be real numeric arrays") from exc
    if lg_h.ndim == 1:
        lg_h = lg_h.reshape(1, -1)
    if h.size == 0 or lf_h.shape != h.shape or lg_h.shape != (h.size, 2):
        raise ValueError("h, lf_h, and lg_h must define one or more two-input CBF rows")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(lf_h)) or not np.all(np.isfinite(lg_h)):
        raise ValueError("CBF rows must be finite")
    return h, lf_h, lg_h


def credible_covariance(filter_covariance, true_error_covariance) -> np.ndarray:
    """Return Pf + [Pm - Pf]_+ for symmetric 4-state covariance matrices."""
    p_filter = _finite_array("filter_covariance", filter_covariance, (4, 4))
    p_tmse = _finite_array("true_error_covariance", true_error_covariance, (4, 4))
    p_filter = 0.5 * (p_filter + p_filter.T)
    p_tmse = 0.5 * (p_tmse + p_tmse.T)
    if np.min(np.linalg.eigvalsh(p_filter)) < -1e-10 or np.min(np.linalg.eigvalsh(p_tmse)) < -1e-10:
        raise ValueError("filter and true-error covariance matrices must be positive semidefinite")
    eigenvalues, eigenvectors = np.linalg.eigh(p_tmse - p_filter)
    positive_part = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    result = p_filter + positive_part
    return 0.5 * (result + result.T)


def credible_geometry_scale(
    *,
    relative_position,
    position_covariance,
    safety_distance,
    confidence_scale,
) -> dict:
    """Map credible relative-position covariance to alpha and lambda."""
    relative_position = _finite_array("relative_position", relative_position, (2,))
    position_covariance = _finite_array("position_covariance", position_covariance, (2, 2))
    position_covariance = 0.5 * (position_covariance + position_covariance.T)
    try:
        safety_distance = float(safety_distance)
        confidence_scale = float(confidence_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("credible geometry parameters must be real numbers") from exc
    if (
        not np.all(np.isfinite([safety_distance, confidence_scale]))
        or safety_distance <= 0.0
        or confidence_scale <= 0.0
        or np.min(np.linalg.eigvalsh(position_covariance)) < -1e-10
    ):
        raise ValueError("credible geometry parameters must be positive and covariance semidefinite")

    distance = float(np.linalg.norm(relative_position))
    if distance <= 0.0:
        raise ValueError("credible geometry requires a nonzero relative position")
    r_hat = relative_position / distance
    r_perp = np.array([-r_hat[1], r_hat[0]], dtype=float)
    e_parallel = confidence_scale * math.sqrt(max(float(r_hat @ position_covariance @ r_hat), 0.0))
    e_perp = confidence_scale * math.sqrt(max(float(r_perp @ position_covariance @ r_perp), 0.0))
    d_min = distance - e_parallel
    if d_min <= safety_distance:
        raise CredibleGeometryDomainError(
            "credible geometry domain requires d_min > safety_distance"
        )
    alpha = math.atan(e_perp / d_min) + math.asin(safety_distance / d_min)
    if not 0.0 < alpha < 0.5 * math.pi:
        raise CredibleGeometryDomainError(
            "credible geometry domain requires 0 < alpha < pi/2"
        )

    chi = (distance * distance - safety_distance * safety_distance) / (safety_distance * safety_distance)
    geometry_scale = (math.cos(alpha) / math.sin(alpha)) ** 2 / chi
    if geometry_scale <= 0.0 or geometry_scale > 1.0 + 1e-10:
        raise ValueError("credible geometry produced lambda outside (0, 1]")
    geometry_scale = min(geometry_scale, 1.0)
    return {
        "lambda": float(geometry_scale),
        "alpha": float(alpha),
        "e_parallel": float(e_parallel),
        "e_perp": float(e_perp),
        "d_min": float(d_min),
    }


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(x, -60.0, 60.0)))))


def colregs_duty(
    *,
    relative_position,
    own_velocity,
    target_velocity,
    d_safe,
    tcpa_horizon,
    kappa_beta,
    kappa_time,
    theta_head,
    theta_limit,
) -> dict:
    """Evaluate the continuous Rules 14-15 right-turn duty for one target."""
    relative_position = _finite_array("relative_position", relative_position, (2,))
    own_velocity = _finite_array("own_velocity", own_velocity, (2,))
    target_velocity = _finite_array("target_velocity", target_velocity, (2,))
    try:
        d_safe, tcpa_horizon, kappa_beta, kappa_time, theta_head, theta_limit = map(
            float,
            (d_safe, tcpa_horizon, kappa_beta, kappa_time, theta_head, theta_limit),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("COLREGs parameters must be real numbers") from exc
    values = np.array([d_safe, tcpa_horizon, kappa_beta, kappa_time, theta_head, theta_limit])
    if (
        not np.all(np.isfinite(values))
        or d_safe <= 0.0
        or tcpa_horizon <= 0.0
        or kappa_beta <= 0.0
        or kappa_time <= 0.0
        or not 0.0 < theta_head < theta_limit < math.pi
    ):
        raise ValueError("invalid COLREGs parameters")

    beta = math.atan2(float(relative_position[1]), float(relative_position[0]))
    psi_head = _sigmoid(kappa_beta * (math.cos(beta) - math.cos(theta_head)))
    psi_crossing = _sigmoid(kappa_beta * (-beta - theta_head)) * _sigmoid(
        kappa_beta * (beta + theta_limit)
    )
    closing_velocity = own_velocity - target_velocity
    speed_squared = float(closing_velocity @ closing_velocity)
    if speed_squared <= 1e-12:
        tcpa = 0.0
        dcpa = float(np.linalg.norm(relative_position))
        risk = 0.0
    else:
        tcpa = float(relative_position @ closing_velocity / speed_squared)
        dcpa = float(np.linalg.norm(relative_position - closing_velocity * tcpa))
        risk = max(0.0, 1.0 - dcpa / d_safe)
        risk *= _sigmoid(kappa_time * tcpa) * _sigmoid(kappa_time * (tcpa_horizon - tcpa))
    duty = risk * (1.0 - (1.0 - psi_head) * (1.0 - psi_crossing))
    return {
        "phi": float(np.clip(duty, 0.0, 1.0)),
        "beta": float(beta),
        "psi_head": float(psi_head),
        "psi_crossing": float(psi_crossing),
        "risk": float(np.clip(risk, 0.0, 1.0)),
        "tcpa": float(tcpa),
        "dcpa": float(dcpa),
    }


def aggregate_colregs_duty(duties) -> float:
    try:
        duties = np.asarray(list(duties), dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("COLREGs duties must be real numbers") from exc
    if not np.all(np.isfinite(duties)) or np.any(duties < 0.0) or np.any(duties > 1.0):
        raise ValueError("COLREGs duties must lie in [0, 1]")
    return float(1.0 - np.prod(1.0 - duties))


def colregs_reference_control(
    *,
    nominal_control,
    aggregate_duty,
    turn_accel,
    yaw_control_gain,
    reference_scale=1.0,
) -> np.ndarray:
    nominal_control = _finite_array("nominal_control", nominal_control, (2,))
    try:
        aggregate_duty = float(aggregate_duty)
        turn_accel = float(turn_accel)
        yaw_control_gain = float(yaw_control_gain)
        reference_scale = float(reference_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("COLREGs reference parameters must be real numbers") from exc
    if (
        not np.all(
            np.isfinite(
                [aggregate_duty, turn_accel, yaw_control_gain, reference_scale]
            )
        )
        or not 0.0 <= aggregate_duty <= 1.0
        or turn_accel <= 0.0
        or yaw_control_gain <= 0.0
        or not 0.0 <= reference_scale <= 1.0
    ):
        raise ValueError("invalid COLREGs reference parameters")
    reference = nominal_control.copy()
    reference[1] -= (
        reference_scale * aggregate_duty * turn_accel / yaw_control_gain
    )
    return reference


def _corecbf_terms(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    surge_accel,
    turn_accel,
    turn_direction,
    usv_params,
    geometry_scale,
    signed_distance_extension,
) -> dict:
    """Evaluate the fixed-direction CoReCBF and its first Lie derivatives."""
    ship_state = _finite_array("ship_state", ship_state, (6,))
    target_position = _finite_array("target_position", target_position, (2,))
    target_velocity = _finite_array("target_velocity", target_velocity, (2,))
    try:
        safety_distance = float(safety_distance)
        surge_accel = float(surge_accel)
        turn_accel = float(turn_accel)
        turn_direction = float(turn_direction)
        m11 = float(usv_params["m11"])
        m22 = float(usv_params["m22"])
        m33 = float(usv_params["m33"])
        x_u = float(usv_params["x_u"])
        y_v = float(usv_params["y_v"])
        n_r = float(usv_params["n_r"])
        yaw_nonlinear = float(usv_params.get("yaw_nonlinear", 10.0))
        geometry_scale = float(geometry_scale)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid CoReCBF or Otter dynamics parameters") from exc
    scalars = np.array(
        [
            safety_distance,
            surge_accel,
            turn_accel,
            m11,
            m22,
            m33,
            x_u,
            y_v,
            n_r,
            yaw_nonlinear,
            geometry_scale,
        ]
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("CoReCBF and Otter dynamics parameters must be finite")
    if safety_distance <= 0.0 or surge_accel <= 0.0 or turn_accel <= 0.0:
        raise ValueError("safety_distance, surge_accel, and turn_accel must be positive")
    if m11 <= 0.0 or m22 <= 0.0 or m33 <= 0.0:
        raise ValueError("m11, m22, and m33 must be positive")
    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be fixed at -1 or 1")
    if not 0.0 < geometry_scale <= 1.0:
        raise ValueError("geometry_scale must lie in (0, 1]")
    turn_direction = int(turn_direction)

    x, y, psi, u, v, r = ship_state
    c, s = math.cos(psi), math.sin(psi)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    rotation_90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    rho = target_position - np.array([x, y])
    distance = float(np.linalg.norm(rho))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("ship and target positions must be distinct")

    e = rho / distance
    n = rotation_90 @ e
    body_velocity = np.array([u, v], dtype=float)
    relative_velocity = rotation @ body_velocity - target_velocity
    v_parallel = float(e @ relative_velocity)
    v_perp = float(n @ relative_velocity)
    delta = distance - safety_distance
    chi = (distance * distance - safety_distance * safety_distance) / (safety_distance * safety_distance)
    h_u = rotation[:, 0]
    surge_projection = float(e @ h_u)
    turn_root = math.sqrt(r * r + math.pi * turn_accel)
    turn_time = math.pi / (turn_root + turn_direction * r)
    v_parallel_positive = max(v_parallel, 0.0)

    turn_distance = delta * abs(delta) if signed_distance_extension else delta * delta
    h = (
        geometry_scale * chi * v_perp * v_perp
        + 2.0 * surge_accel * delta * surge_projection * surge_projection
        + turn_distance / (turn_time * turn_time)
        - v_parallel_positive * v_parallel_positive
    )

    f_nu = np.array(
        [
            (x_u * u + m22 * v * r) / m11,
            (y_v * v - m11 * u * r) / m22,
            (n_r * (1.0 + yaw_nonlinear * abs(r)) * r - (m22 - m11) * u * v) / m33,
        ],
        dtype=float,
    )
    lf_w = rotation @ (f_nu[:2] + r * (rotation_90 @ body_velocity))
    lg_w = rotation @ np.array([[1.0 / m11, 0.0], [0.0, 0.0]], dtype=float)
    lf_r = float(f_nu[2])
    lg_r = np.array([0.0, 1.0 / m33], dtype=float)
    input_coefficient = 2.0 * (
        geometry_scale * chi * v_perp * n - v_parallel_positive * e
    )
    turn_coefficient = (
        2.0
        * turn_direction
        * turn_distance
        / (turn_time * turn_time * turn_root)
    )
    lf_h = (
        2.0
        * (v_parallel_positive - geometry_scale * v_parallel)
        * v_perp
        * v_perp
        / distance
        + float(input_coefficient @ lf_w)
        - 2.0 * surge_accel * v_parallel * surge_projection * surge_projection
        - 4.0
        * surge_accel
        * delta
        * surge_projection
        * float(n @ h_u)
        * (r + v_perp / distance)
        - 2.0
        * (abs(delta) if signed_distance_extension else delta)
        * v_parallel
        / (turn_time * turn_time)
        + turn_coefficient * lf_r
    )
    lg_h = input_coefficient @ lg_w + turn_coefficient * lg_r
    result_values = np.array([h, lf_h, *lg_h, v_parallel, v_perp, turn_time], dtype=float)
    if not np.all(np.isfinite(result_values)):
        raise FloatingPointError("CoReCBF evaluation produced non-finite values")
    return {
        "H": float(h),
        "LfH": float(lf_h),
        "LgH": np.asarray(lg_h, dtype=float).reshape(2),
        "distance": distance,
        "v_parallel": v_parallel,
        "v_perp": v_perp,
        "turn_time": turn_time,
        "geometry_scale": geometry_scale,
    }


def corecbf_terms(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    surge_accel,
    turn_accel,
    turn_direction,
    usv_params,
) -> dict:
    """Evaluate the existing deterministic fixed-direction CoReCBF."""
    return _corecbf_terms(
        ship_state=ship_state,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=safety_distance,
        surge_accel=surge_accel,
        turn_accel=turn_accel,
        turn_direction=turn_direction,
        usv_params=usv_params,
        geometry_scale=1.0,
        signed_distance_extension=False,
    )


def credibility_aware_corecbf_terms(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    surge_accel,
    turn_accel,
    turn_direction,
    usv_params,
    geometry_scale,
) -> dict:
    """Evaluate the credibility-scaled CoReCBF and its first Lie derivatives."""
    return _corecbf_terms(
        ship_state=ship_state,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=safety_distance,
        surge_accel=surge_accel,
        turn_accel=turn_accel,
        turn_direction=turn_direction,
        usv_params=usv_params,
        geometry_scale=geometry_scale,
        signed_distance_extension=True,
    )


def _policy_close_separable_fallback(
    *,
    stage1_decision: np.ndarray,
    input_rows: np.ndarray,
    cbf_lower: np.ndarray,
    nominal_thrusts: np.ndarray,
    thrust_lower: float,
    thrust_upper: float,
    minimum_slack_sum: float,
    lexicographic_tolerance: float,
) -> np.ndarray | None:
    """Project a minimum-slack LP point back toward the policy thrusts."""
    constraint_count = cbf_lower.size
    stage1_thrusts = np.asarray(stage1_decision[:2], dtype=float)
    target_thrusts = np.clip(nominal_thrusts, thrust_lower, thrust_upper)
    if not np.all(np.isfinite(target_thrusts)):
        return None
    slack_limit = minimum_slack_sum + lexicographic_tolerance

    def decision_for(thrusts: np.ndarray) -> np.ndarray | None:
        residuals = input_rows @ thrusts - cbf_lower
        slacks = np.maximum(
            np.zeros(constraint_count, dtype=float), -residuals
        )
        if (
            float(np.sum(slacks)) > slack_limit
            or np.any(residuals + slacks < -1e-10)
        ):
            return None
        return np.concatenate((thrusts, slacks))

    if decision_for(stage1_thrusts) is None:
        return None
    target_decision = decision_for(target_thrusts)
    if target_decision is not None:
        return target_decision

    # Slack sum is convex along this segment.  Bisection therefore finds the
    # policy-nearest feasible point on it without a second numerical solver.
    lower, upper = 0.0, 1.0
    direction = target_thrusts - stage1_thrusts
    for _ in range(32):
        midpoint = 0.5 * (lower + upper)
        candidate = decision_for(stage1_thrusts + midpoint * direction)
        if candidate is None:
            upper = midpoint
        else:
            lower = midpoint
    return decision_for(stage1_thrusts + lower * direction)


def solve_corecbf_qp(
    *,
    nominal_control,
    h,
    lf_h,
    lg_h,
    cbf_gain,
    y_p,
    thrust_lower,
    thrust_upper,
    weights,
    slack_weight,
    eps_abs=1e-7,
    eps_rel=1e-7,
    max_iter=4000,
    polishing=True,
    separable_slack=False,
) -> dict:
    """Solve the shared-slack multi-obstacle QP over the exact twin-thruster set."""
    nominal_control = _finite_array("nominal_control", nominal_control, (2,))
    h, lf_h, lg_h = _finite_cbf_constraints(h, lf_h, lg_h)
    if type(separable_slack) is not bool:
        raise ValueError("separable_slack must be boolean")
    weights = _finite_array("weights", weights, (2,))
    try:
        cbf_gain, y_p, thrust_lower, thrust_upper, slack_weight, eps_abs, eps_rel = map(
            float,
            (cbf_gain, y_p, thrust_lower, thrust_upper, slack_weight, eps_abs, eps_rel),
        )
        max_iter = int(max_iter)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("QP parameters must be real numeric values") from exc
    scalars = np.array(
        [cbf_gain, y_p, thrust_lower, thrust_upper, slack_weight, eps_abs, eps_rel],
        dtype=float,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("QP parameters must be finite")
    if cbf_gain <= 0.0 or y_p <= 0.0 or slack_weight <= 0.0 or np.any(weights <= 0.0):
        raise ValueError("cbf_gain, y_p, slack_weight, and weights must be positive")
    if thrust_lower > thrust_upper or eps_abs <= 0.0 or eps_rel <= 0.0 or max_iter <= 0:
        raise ValueError("invalid thrust bounds or OSQP settings")

    actuator_map = np.array([[1.0, 1.0], [y_p, -y_p]], dtype=float)
    nominal_thrusts = np.array(
        [
            0.5 * (nominal_control[0] + nominal_control[1] / y_p),
            0.5 * (nominal_control[0] - nominal_control[1] / y_p),
        ],
        dtype=float,
    )
    nominal_residuals = lf_h + lg_h @ nominal_control + cbf_gain * h
    tolerance = max(1e-9, 10.0 * eps_abs)
    if (
        np.all(nominal_thrusts >= thrust_lower - tolerance)
        and np.all(nominal_thrusts <= thrust_upper + tolerance)
        and np.all(nominal_residuals >= -tolerance)
    ):
        return {
            "control": nominal_control.copy(),
            "thrusts": nominal_thrusts,
            "residual": float(np.min(nominal_residuals)),
            "residuals": nominal_residuals,
            "relaxed_residuals": nominal_residuals.copy(),
            "slack": 0.0,
            "slacks": np.zeros(h.size, dtype=float),
            "slack_sum": 0.0,
            "slack_positive_count": 0,
            "status": "nominal feasible",
            "solver": "none",
            "iterations": 0,
        }

    try:
        import osqp
        import scipy.sparse as sp
        from scipy.optimize import linprog
    except Exception as exc:
        raise RuntimeError("CoReCBF QP requires osqp and scipy") from exc

    weight_matrix = np.diag(weights)
    cbf_lower = -lf_h - cbf_gain * h
    input_rows = lg_h @ actuator_map
    row_scales = np.maximum(
        1.0,
        np.maximum(np.max(np.abs(input_rows), axis=1), np.abs(cbf_lower)),
    )

    if separable_slack:
        constraint_count = h.size
        stage1 = linprog(
            c=np.concatenate((np.zeros(2), np.ones(constraint_count))),
            A_ub=np.hstack(
                (
                    -input_rows / row_scales[:, None],
                    -np.diag(1.0 / row_scales),
                )
            ),
            b_ub=-cbf_lower / row_scales,
            bounds=[
                (thrust_lower, thrust_upper),
                (thrust_lower, thrust_upper),
                *[(0.0, None)] * constraint_count,
            ],
            method="highs",
        )
        if not stage1.success or stage1.x is None:
            raise RuntimeError(
                f"CoReCBF separable-slack LP failed: {stage1.message}"
            )
        stage1_decision = np.asarray(stage1.x, dtype=float)
        if stage1_decision.shape != (2 + constraint_count,) or not np.all(
            np.isfinite(stage1_decision)
        ):
            raise RuntimeError("CoReCBF separable-slack LP returned an invalid solution")
        stage1_thrusts = stage1_decision[:2]
        stage1_residuals = lf_h + lg_h @ (actuator_map @ stage1_thrusts) + cbf_gain * h
        stage1_slacks = np.maximum.reduce(
            (
                np.zeros(constraint_count, dtype=float),
                stage1_decision[2:],
                -stage1_residuals,
            )
        )
        minimum_slack_sum = float(np.sum(stage1_slacks))
        lexicographic_tolerance = tolerance * max(1, constraint_count)

        thrust_p = actuator_map.T @ weight_matrix @ actuator_map
        thrust_q = -(actuator_map.T @ weight_matrix @ nominal_control)
        objective_scale = max(
            1.0,
            float(np.max(np.abs(thrust_p))),
            float(np.max(np.abs(thrust_q))),
        )
        thrust_p /= objective_scale
        thrust_q /= objective_scale
        p = sp.block_diag(
            (
                sp.csc_matrix(np.triu(thrust_p)),
                sp.csc_matrix((constraint_count, constraint_count)),
            ),
            format="csc",
        )
        q = np.concatenate((thrust_q, np.zeros(constraint_count)))
        thrust_matrix = np.hstack(
            (np.eye(2, dtype=float), np.zeros((2, constraint_count)))
        )
        soft_matrix = np.hstack(
            (
                input_rows / row_scales[:, None],
                np.diag(1.0 / row_scales),
            )
        )
        slack_matrix = np.hstack(
            (np.zeros((constraint_count, 2)), np.eye(constraint_count))
        )
        slack_sum_row = np.concatenate(
            (np.zeros(2), np.ones(constraint_count))
        )[None, :]
        problem = osqp.OSQP()
        problem.setup(
            P=p,
            q=q,
            A=sp.csc_matrix(
                np.vstack(
                    (thrust_matrix, soft_matrix, slack_matrix, slack_sum_row)
                )
            ),
            l=np.concatenate(
                (
                    np.array([thrust_lower, thrust_lower]),
                    cbf_lower / row_scales,
                    np.zeros(constraint_count),
                    np.array([-np.inf]),
                )
            ),
            u=np.concatenate(
                (
                    np.array([thrust_upper, thrust_upper]),
                    np.full(constraint_count, np.inf),
                    np.full(constraint_count, np.inf),
                    np.array([minimum_slack_sum + lexicographic_tolerance]),
                )
            ),
            verbose=False,
            polishing=bool(polishing),
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            rho=10.0,
            alpha=1.9,
        )
        problem.warm_start(x=np.concatenate((stage1_thrusts, stage1_slacks)))
        stable_fallback_decision = None
        fallback_checked = False

        def fallback_decision() -> tuple[np.ndarray, str]:
            nonlocal stable_fallback_decision, fallback_checked
            if not fallback_checked:
                stable_fallback_decision = _policy_close_separable_fallback(
                    stage1_decision=stage1_decision,
                    input_rows=input_rows,
                    cbf_lower=cbf_lower,
                    nominal_thrusts=nominal_thrusts,
                    thrust_lower=thrust_lower,
                    thrust_upper=thrust_upper,
                    minimum_slack_sum=minimum_slack_sum,
                    lexicographic_tolerance=lexicographic_tolerance,
                )
                fallback_checked = True
            if stable_fallback_decision is not None:
                return stable_fallback_decision.copy(), "stable fallback"
            return np.concatenate((stage1_thrusts, stage1_slacks)), "feasible fallback"

        result = problem.solve(raise_error=False)
        status = str(getattr(result.info, "status", "")).lower()
        if status in {"solved", "solved inaccurate"} and result.x is not None:
            decision = np.asarray(result.x, dtype=float)
        else:
            decision, fallback_status = fallback_decision()
            status = f"{status or 'stage 2 failed'} {fallback_status}"
        if decision.shape != (2 + constraint_count,) or not np.all(
            np.isfinite(decision)
        ):
            raise RuntimeError("CoReCBF separable-slack QP returned an invalid solution")
        raw_thrusts = decision[:2]
        raw_slacks = decision[2:]
        if np.any(raw_slacks < -lexicographic_tolerance):
            decision, fallback_status = fallback_decision()
            raw_thrusts = decision[:2]
            raw_slacks = decision[2:]
            status = f"{status} {fallback_status}"
        thrusts = np.clip(raw_thrusts, thrust_lower, thrust_upper)
        control = actuator_map @ thrusts
        residuals = lf_h + lg_h @ control + cbf_gain * h
        slacks = np.maximum.reduce(
            (np.zeros(constraint_count, dtype=float), raw_slacks, -residuals)
        )
        slack_sum = float(np.sum(slacks))
        if slack_sum > minimum_slack_sum + 2.0 * lexicographic_tolerance:
            decision, fallback_status = fallback_decision()
            thrusts = decision[:2]
            control = actuator_map @ thrusts
            residuals = lf_h + lg_h @ control + cbf_gain * h
            slacks = np.maximum.reduce(
                (np.zeros(constraint_count, dtype=float), decision[2:], -residuals)
            )
            slack_sum = float(np.sum(slacks))
            status = f"{status} {fallback_status}"
        relaxed_residuals = residuals + slacks
        if np.any(relaxed_residuals < -1e-10):
            raise RuntimeError(
                "CoReCBF separable-slack projection failed relaxed verification"
            )
        if np.any(thrusts < thrust_lower - tolerance) or np.any(
            thrusts > thrust_upper + tolerance
        ):
            raise RuntimeError("CoReCBF separable-slack QP failed thrust verification")
        return {
            "control": control,
            "thrusts": thrusts,
            "residual": float(np.min(residuals)),
            "residuals": residuals,
            "relaxed_residuals": relaxed_residuals,
            "slack": float(np.max(slacks, initial=0.0)),
            "slacks": slacks,
            "slack_sum": slack_sum,
            "slack_positive_count": int(
                np.count_nonzero(slacks > 10.0 * lexicographic_tolerance)
            ),
            "status": f"{status} separable slack",
            "solver": "osqp",
            "iterations": int(getattr(result.info, "iter", 0) or 0),
        }

    # Lexicographic stage 1: find the minimum common CBF violation permitted by
    # the exact twin-thruster bounds. The configured weight is retained for API
    # compatibility; any positive multiplier has the same stage-1 minimizer.
    stage1 = linprog(
        c=np.array([0.0, 0.0, slack_weight / max(1.0, slack_weight)]),
        A_ub=np.column_stack(
            (-input_rows / row_scales[:, None], -1.0 / row_scales)
        ),
        b_ub=-cbf_lower / row_scales,
        bounds=[
            (thrust_lower, thrust_upper),
            (thrust_lower, thrust_upper),
            (0.0, None),
        ],
        method="highs",
    )
    if not stage1.success or stage1.x is None:
        raise RuntimeError(f"CoReCBF minimum-slack LP failed: {stage1.message}")
    stage1_decision = np.asarray(stage1.x, dtype=float)
    stage1_thrusts = stage1_decision[:2]
    stage1_control = actuator_map @ stage1_thrusts
    stage1_residuals = lf_h + lg_h @ stage1_control + cbf_gain * h
    minimum_slack = max(
        0.0,
        float(stage1_decision[2]),
        float(-np.min(stage1_residuals)),
    )

    # Lexicographic stage 2: with minimum slack fixed (up to solver tolerance),
    # choose the closest admissible control to the policy command.
    lexicographic_tolerance = tolerance
    p = actuator_map.T @ weight_matrix @ actuator_map
    q = -(actuator_map.T @ weight_matrix @ nominal_control)
    objective_scale = max(1.0, float(np.max(np.abs(p))), float(np.max(np.abs(q))))
    p /= objective_scale
    q /= objective_scale
    constraint_matrix = sp.csc_matrix(
        np.vstack(
            (
                np.eye(2, dtype=float),
                input_rows / row_scales[:, None],
            )
        )
    )
    lower = np.concatenate(
        (
            np.array([thrust_lower, thrust_lower]),
            (cbf_lower - minimum_slack - lexicographic_tolerance) / row_scales,
        )
    )
    upper = np.concatenate((np.array([thrust_upper, thrust_upper]), np.full(h.size, np.inf)))
    problem = osqp.OSQP()
    problem.setup(
        P=sp.csc_matrix(np.triu(p)),
        q=q,
        A=constraint_matrix,
        l=lower,
        u=upper,
        verbose=False,
        polishing=bool(polishing),
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        rho=10.0,
        alpha=1.9,
    )
    problem.warm_start(x=stage1_thrusts)
    result = problem.solve(raise_error=False)
    status = str(getattr(result.info, "status", "")).lower()
    if status == "maximum iterations reached":
        decision = stage1_thrusts
        status = "maximum iterations reached feasible fallback"
    elif status in {"solved", "solved inaccurate"} and result.x is not None:
        decision = np.asarray(result.x, dtype=float)
    else:
        decision = stage1_thrusts
        status = f"{status or 'stage 2 failed'} feasible fallback"
    if decision.shape != (2,) or not np.all(np.isfinite(decision)):
        raise RuntimeError("CoReCBF QP returned an invalid solution")
    raw_thrusts = decision
    raw_control = actuator_map @ raw_thrusts
    raw_residuals = lf_h + lg_h @ raw_control + cbf_gain * h
    raw_slack = max(minimum_slack, 0.0, float(-np.min(raw_residuals)))
    raw_relaxed_residuals = raw_residuals + raw_slack
    residual_scales = np.maximum(
        1.0,
        np.abs(lf_h)
        + np.abs(cbf_gain * h)
        + np.abs(lg_h) @ np.abs(raw_control)
        + abs(raw_slack),
    )
    thrust_tolerance = tolerance * max(1.0, abs(thrust_lower), abs(thrust_upper))
    if (
        np.any(raw_thrusts < thrust_lower - thrust_tolerance)
        or np.any(raw_thrusts > thrust_upper + thrust_tolerance)
    ):
        worst = int(np.argmin(raw_relaxed_residuals / residual_scales))
        raise RuntimeError(
            "CoReCBF QP solution failed bound verification: "
            f"status={status}, row={worst}, residual={raw_residuals[worst]:.9g}, "
            f"slack={raw_slack:.9g}, relaxed={raw_relaxed_residuals[worst]:.9g}, "
            f"scale={residual_scales[worst]:.9g}, tolerance={tolerance:.9g}, "
            f"thrusts={raw_thrusts.tolist()}, bounds={[thrust_lower, thrust_upper]}, "
            f"nominal={nominal_control.tolist()}, h={h.tolist()}, lf_h={lf_h.tolist()}, "
            f"lg_h={lg_h.tolist()}"
        )
    thrusts = np.clip(raw_thrusts, thrust_lower, thrust_upper)
    control = actuator_map @ thrusts
    residuals = lf_h + lg_h @ control + cbf_gain * h
    slack = max(0.0, raw_slack, float(-np.min(residuals)))
    relaxed_residuals = residuals + slack
    if np.any(relaxed_residuals < -1e-10):
        raise RuntimeError("CoReCBF QP projection failed to satisfy the relaxed constraints")
    if not np.array_equal(thrusts, raw_thrusts) or slack != minimum_slack:
        status = f"{status} projected"
    return {
        "control": control,
        "thrusts": thrusts,
        "residual": float(np.min(residuals)),
        "residuals": residuals,
        "relaxed_residuals": relaxed_residuals,
        "slack": slack,
        "slacks": np.full(h.size, slack, dtype=float),
        "slack_sum": float(slack * h.size),
        "slack_positive_count": int(h.size if slack > tolerance else 0),
        "status": status,
        "solver": "osqp",
        "iterations": int(getattr(result.info, "iter", 0) or 0),
    }
