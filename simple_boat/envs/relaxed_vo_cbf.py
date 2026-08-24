import math

import numpy as np


class RelaxedVOCBFDomainError(ValueError):
    """The direct VO barrier is outside its distance domain."""


class RelaxedVOCBFNoVerifiedAction(RuntimeError):
    """The QP produced no independently verified physical action."""


def _finite_vector(name: str, value, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric vector") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-({size},) vector")
    return array


def _positive_scalar(name: str, value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def otter_acceleration_maps(
    *,
    m11,
    m33,
    y_p,
    thrust_lower,
    thrust_upper,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    m11 = _positive_scalar("m11", m11)
    m33 = _positive_scalar("m33", m33)
    y_p = _positive_scalar("y_p", y_p)
    try:
        thrust_lower = float(thrust_lower)
        thrust_upper = float(thrust_upper)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("thrust bounds must be real numbers") from exc
    if (
        not np.all(np.isfinite([thrust_lower, thrust_upper]))
        or not thrust_lower < 0.0 < thrust_upper
    ):
        raise ValueError("thrust bounds must be finite with lower < 0 < upper")

    surge_scale = min(
        2.0 * thrust_upper / m11,
        -2.0 * thrust_lower / m11,
    )
    yaw_scale = y_p * (thrust_upper - thrust_lower) / m33
    yaw_normalizer = surge_scale / yaw_scale
    if (
        not np.all(
            np.isfinite([surge_scale, yaw_scale, yaw_normalizer])
        )
        or min(surge_scale, yaw_scale, yaw_normalizer) <= 0.0
    ):
        raise ValueError("Otter acceleration scales must be positive and finite")

    thrust_to_control = np.array(
        [
            [1.0 / m11, 1.0 / m11],
            [
                yaw_normalizer * y_p / m33,
                -yaw_normalizer * y_p / m33,
            ],
        ],
        dtype=float,
    )
    control_to_thrust = np.array(
        [
            [
                0.5 * m11,
                0.5 * m33 / y_p / yaw_normalizer,
            ],
            [
                0.5 * m11,
                -0.5 * m33 / y_p / yaw_normalizer,
            ],
        ],
        dtype=float,
    )
    if not np.all(
        np.isfinite(np.concatenate((thrust_to_control, control_to_thrust)))
    ):
        raise ValueError("Otter acceleration maps must be finite")
    return (
        thrust_to_control,
        control_to_thrust,
        float(surge_scale),
        float(yaw_scale),
    )


def time_to_collision(
    relative_position,
    relative_velocity,
    safety_distance,
) -> float:
    position = _finite_vector("relative_position", relative_position, 2)
    velocity = _finite_vector("relative_velocity", relative_velocity, 2)
    radius = _positive_scalar("safety_distance", safety_distance)
    a = float(velocity @ velocity)
    if a <= 1e-12:
        return math.inf
    b = 2.0 * float(position @ velocity)
    c = float(position @ position) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return math.inf
    root = math.sqrt(max(discriminant, 0.0))
    roots = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    positive = [
        value for value in roots if value > 1e-12 and math.isfinite(value)
    ]
    return float(min(positive)) if positive else math.inf


def ma_cbf_vo_reference_terms(
    *,
    relative_position,
    relative_velocity,
    safety_distance,
    max_accel,
) -> dict:
    """Paper double-integrator VO and braking-CBF rows, without Otter dynamics."""
    position = _finite_vector("relative_position", relative_position, 2)
    velocity = _finite_vector("relative_velocity", relative_velocity, 2)
    radius = _positive_scalar("safety_distance", safety_distance)
    acceleration = _positive_scalar("max_accel", max_accel)
    distance = float(np.linalg.norm(position))
    if distance <= radius:
        raise RelaxedVOCBFDomainError(
            "paper VO reference requires distance > safety_distance"
        )

    collision_time = time_to_collision(position, velocity, radius)
    vo = None
    if math.isfinite(collision_time):
        speed = float(np.linalg.norm(velocity))
        cone_root = math.sqrt(distance * distance - radius * radius)
        velocity_gradient = position + cone_root * velocity / speed
        vo = {
            "H": float(position @ velocity + cone_root * speed),
            "LfH": float(
                speed * speed + speed * float(position @ velocity) / cone_root
            ),
            "LgH_u": -velocity_gradient,
            "weight": 1.0 / collision_time,
        }

    direction = position / distance
    radial_speed = float(velocity @ direction)
    hard = None
    if radial_speed < 0.0:
        radial_drift = float(
            (velocity @ velocity - radial_speed * radial_speed) / distance
        )
        hard = {
            "H": float(
                distance - radius - radial_speed * radial_speed / (2.0 * acceleration)
            ),
            "LfH": float(
                radial_speed - radial_speed * radial_drift / acceleration
            ),
            "LgH_u": radial_speed * direction / acceleration,
        }
    return {"ttc": float(collision_time), "vo": vo, "hard": hard}


def _relative_acceleration_affine(
    *,
    ship_state,
    target_position,
    target_velocity,
    usv_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ship = _finite_vector("ship_state", ship_state, 6)
    target_position = _finite_vector("target_position", target_position, 2)
    target_velocity = _finite_vector("target_velocity", target_velocity, 2)
    try:
        m11 = float(usv_params["m11"])
        m22 = float(usv_params["m22"])
        x_u = float(usv_params["x_u"])
        y_v = float(usv_params["y_v"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "invalid Otter translational dynamics parameters"
        ) from exc
    if (
        not np.all(np.isfinite([m11, m22, x_u, y_v]))
        or m11 <= 0.0
        or m22 <= 0.0
    ):
        raise ValueError(
            "Otter translational dynamics parameters must be finite "
            "with positive masses"
        )

    _, _, psi, u, v, r = ship
    c, s = math.cos(psi), math.sin(psi)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    rotation_90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    body_velocity = np.array([u, v], dtype=float)
    body_drift = np.array(
        [
            (x_u * u + m22 * v * r) / m11,
            (y_v * v - m11 * u * r) / m22,
        ],
        dtype=float,
    )
    own_velocity = rotation @ body_velocity
    own_acceleration_drift = rotation @ (
        body_drift + r * (rotation_90 @ body_velocity)
    )
    relative_position = ship[:2] - target_position
    relative_velocity = own_velocity - target_velocity
    relative_acceleration_drift = own_acceleration_drift
    relative_acceleration_gain = rotation[:, 0] / m11
    return (
        relative_position,
        relative_velocity,
        relative_acceleration_drift,
        relative_acceleration_gain,
    )


def vo_barrier_value(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
) -> float:
    ship = _finite_vector("ship_state", ship_state, 6)
    target_position = _finite_vector(
        "target_position", target_position, 2
    )
    target_velocity = _finite_vector(
        "target_velocity", target_velocity, 2
    )
    radius = _positive_scalar("safety_distance", safety_distance)
    _, _, psi, u, v, _ = ship
    c, s = math.cos(psi), math.sin(psi)
    position = ship[:2] - target_position
    velocity = np.array([c * u - s * v, s * u + c * v]) - target_velocity
    distance = float(np.linalg.norm(position))
    speed = float(np.linalg.norm(velocity))
    if distance <= radius or speed <= 0.0:
        raise RelaxedVOCBFDomainError(
            "predictive VO barrier requires distance > safety_distance "
            "and positive relative speed"
        )
    return float(
        position @ velocity
        + math.sqrt(distance * distance - radius * radius) * speed
    )


def hard_collision_barrier_value(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    braking_accel,
) -> float:
    ship = _finite_vector("ship_state", ship_state, 6)
    target_position = _finite_vector(
        "target_position", target_position, 2
    )
    target_velocity = _finite_vector(
        "target_velocity", target_velocity, 2
    )
    radius = _positive_scalar("safety_distance", safety_distance)
    braking_accel = _positive_scalar("braking_accel", braking_accel)
    _, _, psi, u, v, _ = ship
    c, s = math.cos(psi), math.sin(psi)
    position = ship[:2] - target_position
    distance = float(np.linalg.norm(position))
    if distance <= 0.0:
        raise ValueError("hard collision barrier requires positive distance")
    velocity = np.array([c * u - s * v, s * u + c * v]) - target_velocity
    radial_speed = float(velocity @ (position / distance))
    closing_speed = max(0.0, -radial_speed)
    return float(
        distance
        - radius
        - closing_speed * closing_speed / (2.0 * braking_accel)
    )


def relaxed_vo_cbf_terms(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    usv_params,
) -> dict | None:
    radius = _positive_scalar("safety_distance", safety_distance)
    position, velocity, acceleration_drift, acceleration_gain = (
        _relative_acceleration_affine(
            ship_state=ship_state,
            target_position=target_position,
            target_velocity=target_velocity,
            usv_params=usv_params,
        )
    )
    distance = float(np.linalg.norm(position))
    if distance <= radius:
        raise RelaxedVOCBFDomainError(
            "Relaxed VO-CBF domain requires distance > safety_distance"
        )
    collision_time = time_to_collision(position, velocity, radius)
    if not math.isfinite(collision_time):
        return None

    speed = float(np.linalg.norm(velocity))
    cone_root = math.sqrt(distance * distance - radius * radius)
    velocity_gradient = position + cone_root * velocity / speed
    h = vo_barrier_value(
        ship_state=ship_state,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=radius,
    )
    lf_h = float(
        speed * speed
        + speed * float(position @ velocity) / cone_root
        + velocity_gradient @ acceleration_drift
    )
    lg_tau_u = float(velocity_gradient @ acceleration_gain)
    values = np.array([h, lf_h, lg_tau_u, collision_time], dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "Relaxed VO-CBF evaluation produced non-finite values"
        )
    return {
        "H": h,
        "LfH": lf_h,
        "LgH_tau": np.array([lg_tau_u, 0.0], dtype=float),
        "LgH_thrust": np.array([lg_tau_u, lg_tau_u], dtype=float),
        "ttc": collision_time,
        "distance": distance,
        "relative_speed": speed,
    }


def hard_collision_cbf_terms(
    *,
    ship_state,
    target_position,
    target_velocity,
    safety_distance,
    braking_accel,
    usv_params,
) -> dict | None:
    radius = _positive_scalar("safety_distance", safety_distance)
    braking_accel = _positive_scalar("braking_accel", braking_accel)
    position, velocity, acceleration_drift, acceleration_gain = (
        _relative_acceleration_affine(
            ship_state=ship_state,
            target_position=target_position,
            target_velocity=target_velocity,
            usv_params=usv_params,
        )
    )
    distance = float(np.linalg.norm(position))
    if distance <= 1e-12:
        raise ValueError("hard collision CBF requires positive distance")
    direction = position / distance
    radial_speed = float(velocity @ direction)
    closing_speed = max(0.0, -radial_speed)
    if closing_speed <= 0.0:
        return None

    speed_squared = float(velocity @ velocity)
    radial_drift = float(
        direction @ acceleration_drift
        + (speed_squared - radial_speed * radial_speed) / distance
    )
    radial_gain = float(direction @ acceleration_gain)
    h = hard_collision_barrier_value(
        ship_state=ship_state,
        target_position=target_position,
        target_velocity=target_velocity,
        safety_distance=radius,
        braking_accel=braking_accel,
    )
    lf_h = float(
        radial_speed + closing_speed * radial_drift / braking_accel
    )
    lg_tau_u = float(closing_speed * radial_gain / braking_accel)
    values = np.array([h, lf_h, lg_tau_u], dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "hard collision CBF evaluation produced non-finite values"
        )
    return {
        "H": h,
        "LfH": lf_h,
        "LgH_tau": np.array([lg_tau_u, 0.0], dtype=float),
        "LgH_thrust": np.array([lg_tau_u, lg_tau_u], dtype=float),
        "distance": distance,
        "closing_speed": closing_speed,
    }


def _finite_rows(
    name: str, h, lf_h, lg_h
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        h_array = np.asarray(h, dtype=float).reshape(-1)
        lf_array = np.asarray(lf_h, dtype=float).reshape(-1)
        lg_array = np.asarray(lg_h, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} rows must be real numeric arrays") from exc
    if lf_array.shape != h_array.shape or lg_array.shape != (
        h_array.size,
        2,
    ):
        raise ValueError(f"{name} rows have inconsistent shapes")
    if (
        not np.all(np.isfinite(h_array))
        or not np.all(np.isfinite(lf_array))
        or not np.all(np.isfinite(lg_array))
    ):
        raise ValueError(f"{name} rows must be finite")
    return h_array, lf_array, lg_array


def _finite_affine_rows(name, offset, rows) -> tuple[np.ndarray, np.ndarray]:
    try:
        offset = np.asarray(offset, dtype=float).reshape(-1)
        rows = np.asarray(rows, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} affine rows must be real numeric arrays") from exc
    if rows.shape != (offset.size, 2):
        raise ValueError(f"{name} affine rows have inconsistent shapes")
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} affine rows must be finite")
    return offset, rows


def solve_relaxed_affine_qp(
    *,
    nominal_thrusts,
    control_to_thrust,
    vo_offset,
    vo_input_rows,
    vo_slack_ids,
    vo_ttc,
    hard_offset,
    hard_input_rows,
    k_u,
    k_vo,
    thrust_lower,
    thrust_upper,
    thrust_sum_lower=None,
    thrust_sum_upper=None,
    eps_abs=1e-7,
    eps_rel=1e-7,
    max_iter=4000,
    polishing=True,
) -> dict:
    nominal = _finite_vector("nominal_thrusts", nominal_thrusts, 2)
    try:
        control_to_thrust = np.asarray(
            control_to_thrust, dtype=float
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "control_to_thrust must be a real numeric matrix"
        ) from exc
    if (
        control_to_thrust.shape != (2, 2)
        or not np.all(np.isfinite(control_to_thrust))
    ):
        raise ValueError(
            "control_to_thrust must be a finite shape-(2, 2) matrix"
        )
    try:
        thrust_to_control = np.linalg.inv(control_to_thrust)
    except np.linalg.LinAlgError as exc:
        raise ValueError("control_to_thrust must be invertible") from exc
    if not np.all(np.isfinite(thrust_to_control)):
        raise ValueError("control_to_thrust must be invertible")

    vo_offset, vo_input_rows = _finite_affine_rows(
        "VO", vo_offset, vo_input_rows
    )
    hard_offset, hard_input_rows = _finite_affine_rows(
        "hard CBF", hard_offset, hard_input_rows
    )
    try:
        vo_ttc = np.asarray(vo_ttc, dtype=float).reshape(-1)
        raw_slack_ids = np.asarray(vo_slack_ids)
        (
            k_u,
            k_vo,
            thrust_lower,
            thrust_upper,
            eps_abs,
            eps_rel,
        ) = map(
            float,
            (
                k_u,
                k_vo,
                thrust_lower,
                thrust_upper,
                eps_abs,
                eps_rel,
            ),
        )
        max_iter = int(max_iter)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Relaxed VO-CBF QP parameters must be numeric"
        ) from exc
    scalars = np.array(
        [
            k_u,
            k_vo,
            thrust_lower,
            thrust_upper,
            eps_abs,
            eps_rel,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("Relaxed VO-CBF QP parameters must be finite")
    if (thrust_sum_lower is None) != (thrust_sum_upper is None):
        raise ValueError(
            "total-thrust bounds must either both be provided or both be omitted"
        )
    if thrust_sum_lower is not None:
        try:
            thrust_sum_lower = float(thrust_sum_lower)
            thrust_sum_upper = float(thrust_sum_upper)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("total-thrust bounds must be numeric") from exc
        if (
            not math.isfinite(thrust_sum_lower)
            or not math.isfinite(thrust_sum_upper)
            or thrust_sum_lower > thrust_sum_upper
        ):
            raise ValueError("total-thrust bounds must be finite and ordered")
    if raw_slack_ids.ndim != 1 or raw_slack_ids.shape != vo_offset.shape:
        raise ValueError("every VO row requires one slack id")
    if not np.issubdtype(raw_slack_ids.dtype, np.integer):
        raise ValueError("VO slack ids must be integers")
    vo_slack_ids = raw_slack_ids.astype(int, copy=False)
    if not np.all(np.isfinite(vo_ttc)) or np.any(vo_ttc <= 0.0):
        raise ValueError("every VO slack requires a finite positive TTC")
    if vo_offset.size:
        expected_ids = np.arange(vo_ttc.size, dtype=int)
        if (
            vo_ttc.size == 0
            or np.any(vo_slack_ids < 0)
            or np.any(vo_slack_ids >= vo_ttc.size)
            or not np.array_equal(np.unique(vo_slack_ids), expected_ids)
        ):
            raise ValueError("VO slack ids must cover every declared slack")
    elif vo_ttc.size or vo_slack_ids.size:
        raise ValueError("empty VO rows require empty slack ids and TTCs")
    if (
        k_u <= 0.0
        or k_vo <= 0.0
        or thrust_lower > thrust_upper
        or eps_abs <= 0.0
        or eps_rel <= 0.0
        or max_iter <= 0
    ):
        raise ValueError(
            "invalid Relaxed VO-CBF gains, weights, bounds, "
            "or OSQP settings"
        )

    nominal_control = thrust_to_control @ nominal
    vo_input_control = vo_input_rows @ control_to_thrust
    hard_input_control = hard_input_rows @ control_to_thrust
    vo_raw_nominal = vo_offset + vo_input_rows @ nominal
    hard_nominal = hard_offset + hard_input_rows @ nominal
    tolerance = max(1e-9, 10.0 * eps_abs)
    within_bounds = bool(
        np.all(nominal >= thrust_lower - tolerance)
        and np.all(nominal <= thrust_upper + tolerance)
        and (
            thrust_sum_lower is None
            or (
                thrust_sum_lower - tolerance
                <= float(np.sum(nominal))
                <= thrust_sum_upper + tolerance
            )
        )
    )
    safety_rows = np.vstack((vo_input_control, hard_input_control))
    rank = (
        int(np.linalg.matrix_rank(safety_rows, tol=1e-10))
        if safety_rows.size
        else 0
    )
    yaw_sensitivity = (
        float(np.max(np.abs(safety_rows[:, 1])))
        if safety_rows.size
        else 0.0
    )
    if (
        within_bounds
        and np.all(vo_raw_nominal >= -tolerance)
        and np.all(hard_nominal >= -tolerance)
    ):
        return {
            "thrusts": np.clip(nominal, thrust_lower, thrust_upper),
            "control": nominal_control,
            "slacks": np.zeros(vo_ttc.size, dtype=float),
            "eta_slacks": np.zeros(vo_ttc.size, dtype=float),
            "vo_residuals": vo_raw_nominal,
            "relaxed_vo_residuals": vo_raw_nominal.copy(),
            "hard_residuals": hard_nominal,
            "status": "nominal feasible",
            "solver": "none",
            "iterations": 0,
            "input_rows": safety_rows,
            "safety_row_rank": rank,
            "yaw_sensitivity_max": yaw_sensitivity,
        }

    try:
        import osqp
        import scipy.sparse as sp
    except Exception as exc:
        raise RuntimeError(
            "Relaxed VO-CBF QP requires the existing osqp and scipy packages"
        ) from exc

    n_vo_rows = int(vo_offset.size)
    n_slacks = int(vo_ttc.size)
    n_variables = 2 + n_slacks
    lambda_per_eta = np.sqrt(vo_ttc / k_vo)
    if (
        not np.all(np.isfinite(lambda_per_eta))
        or np.any(lambda_per_eta <= 0.0)
    ):
        raise ValueError(
            "Relaxed VO-CBF TTC scaling must be positive and finite"
        )
    p_diagonal = 2.0 * np.concatenate(
        (np.array([k_u, k_u]), np.ones(n_slacks, dtype=float))
    )
    q = np.concatenate(
        (-2.0 * k_u * nominal_control, np.zeros(n_slacks))
    )
    objective_scale = max(
        1.0,
        float(np.max(np.abs(p_diagonal))),
        float(np.max(np.abs(q))),
    )
    p_diagonal /= objective_scale
    q /= objective_scale

    blocks = [
        np.column_stack(
            (
                control_to_thrust,
                np.zeros((2, n_slacks), dtype=float),
            )
        )
    ]
    lower = [np.full(2, thrust_lower, dtype=float)]
    upper = [np.full(2, thrust_upper, dtype=float)]
    if n_slacks:
        blocks.append(
            np.column_stack(
                (
                    np.zeros((n_slacks, 2), dtype=float),
                    np.eye(n_slacks),
                )
            )
        )
        lower.append(np.zeros(n_slacks, dtype=float))
        upper.append(np.full(n_slacks, np.inf))
    if n_vo_rows:
        slack_matrix = np.zeros((n_vo_rows, n_slacks), dtype=float)
        slack_matrix[
            np.arange(n_vo_rows), vo_slack_ids
        ] = lambda_per_eta[vo_slack_ids]
        blocks.append(
            np.column_stack(
                (vo_input_control, slack_matrix)
            )
        )
        lower.append(-vo_offset)
        upper.append(np.full(n_vo_rows, np.inf))
    if thrust_sum_lower is not None:
        blocks.append(
            np.column_stack(
                (
                    np.sum(control_to_thrust, axis=0, keepdims=True),
                    np.zeros((1, n_slacks), dtype=float),
                )
            )
        )
        lower.append(np.array([thrust_sum_lower], dtype=float))
        upper.append(np.array([thrust_sum_upper], dtype=float))
    if hard_offset.size:
        blocks.append(
            np.column_stack(
                (
                    hard_input_control,
                    np.zeros((hard_offset.size, n_slacks), dtype=float),
                )
            )
        )
        lower.append(-hard_offset)
        upper.append(np.full(hard_offset.size, np.inf))

    problem = osqp.OSQP()
    problem.setup(
        P=sp.diags(p_diagonal, format="csc"),
        q=q,
        A=sp.csc_matrix(np.vstack(blocks)),
        l=np.concatenate(lower),
        u=np.concatenate(upper),
        verbose=False,
        polishing=bool(polishing),
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
    )
    warm_lambda = np.zeros(n_slacks, dtype=float)
    if n_vo_rows:
        np.maximum.at(warm_lambda, vo_slack_ids, -vo_raw_nominal)
        warm_lambda = np.maximum(warm_lambda, 0.0)
    problem.warm_start(
        x=np.concatenate(
            (
                nominal_control,
                warm_lambda / lambda_per_eta,
            )
        )
    )
    result = problem.solve(raise_error=False)
    info = getattr(result, "info", None)
    status = str(getattr(info, "status", "")).lower()
    if status not in {"solved", "solved inaccurate"}:
        raise RelaxedVOCBFNoVerifiedAction(
            f"Relaxed VO-CBF QP rejected solver status: "
            f"{status or 'missing solver status'}"
        )
    candidate = getattr(result, "x", None)
    if candidate is None:
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP returned no candidate decision"
        )

    try:
        decision = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP returned a malformed decision vector"
        ) from exc
    if decision.shape != (n_variables,) or not np.all(
        np.isfinite(decision)
    ):
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP returned an invalid decision vector"
        )
    raw_control = decision[:2]
    raw_eta = decision[2:]
    raw_thrusts = control_to_thrust @ raw_control
    raw_slacks = raw_eta * lambda_per_eta
    bound_tolerance = tolerance * max(
        1.0, abs(thrust_lower), abs(thrust_upper)
    )
    if (
        np.any(raw_thrusts < thrust_lower - bound_tolerance)
        or np.any(raw_thrusts > thrust_upper + bound_tolerance)
        or np.any(raw_eta < -tolerance)
    ):
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP solution failed bound verification"
        )
    if thrust_sum_lower is not None and not (
        thrust_sum_lower - bound_tolerance
        <= float(np.sum(raw_thrusts))
        <= thrust_sum_upper + bound_tolerance
    ):
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP solution failed total-thrust verification"
        )

    thrusts = np.clip(raw_thrusts, thrust_lower, thrust_upper)
    control = thrust_to_control @ thrusts
    slacks = np.maximum(raw_slacks, 0.0)
    eta_slacks = np.maximum(raw_eta, 0.0)
    vo_residuals = vo_offset + vo_input_rows @ thrusts
    relaxed_vo_residuals = vo_residuals + slacks[vo_slack_ids]
    hard_residuals = hard_offset + hard_input_rows @ thrusts
    if np.any(relaxed_vo_residuals < -tolerance):
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP failed relaxed-VO residual verification"
        )
    if np.any(hard_residuals < -tolerance):
        raise RelaxedVOCBFNoVerifiedAction(
            "Relaxed VO-CBF QP failed hard-CBF residual verification"
        )
    return {
        "thrusts": thrusts,
        "control": control,
        "slacks": slacks,
        "eta_slacks": eta_slacks,
        "vo_residuals": vo_residuals,
        "relaxed_vo_residuals": relaxed_vo_residuals,
        "hard_residuals": hard_residuals,
        "status": status,
        "solver": "osqp",
        "iterations": int(getattr(info, "iter", 0) or 0),
        "input_rows": safety_rows,
        "safety_row_rank": rank,
        "yaw_sensitivity_max": yaw_sensitivity,
    }


def solve_relaxed_vo_cbf_qp(
    *,
    nominal_thrusts,
    control_to_thrust,
    vo_h,
    vo_lf_h,
    vo_lg_thrust,
    vo_ttc,
    hard_h,
    hard_lf_h,
    hard_lg_thrust,
    alpha_vo,
    alpha_c,
    k_u,
    k_vo,
    thrust_lower,
    thrust_upper,
    eps_abs=1e-7,
    eps_rel=1e-7,
    max_iter=4000,
    polishing=True,
) -> dict:
    vo_h, vo_lf_h, vo_lg_thrust = _finite_rows(
        "VO", vo_h, vo_lf_h, vo_lg_thrust
    )
    hard_h, hard_lf_h, hard_lg_thrust = _finite_rows(
        "hard CBF", hard_h, hard_lf_h, hard_lg_thrust
    )
    alpha_vo = _positive_scalar("alpha_vo", alpha_vo)
    alpha_c = _positive_scalar("alpha_c", alpha_c)
    return solve_relaxed_affine_qp(
        nominal_thrusts=nominal_thrusts,
        control_to_thrust=control_to_thrust,
        vo_offset=vo_lf_h + alpha_vo * vo_h,
        vo_input_rows=vo_lg_thrust,
        vo_slack_ids=np.arange(vo_h.size, dtype=int),
        vo_ttc=vo_ttc,
        hard_offset=hard_lf_h + alpha_c * hard_h,
        hard_input_rows=hard_lg_thrust,
        k_u=k_u,
        k_vo=k_vo,
        thrust_lower=thrust_lower,
        thrust_upper=thrust_upper,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polishing=polishing,
    )
