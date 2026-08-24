# envs/dynamics.py — Otter USV 3-DOF dynamics model
import numpy as np


def _clip(val, v_min, v_max):
    return np.clip(val, v_min, v_max)


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def prop_thrust(n, k_pos, k_neg):
    """Quadratic propeller thrust: T = k * n * |n|."""
    if n >= 0:
        return k_pos * n * abs(n)
    else:
        return k_neg * n * abs(n)


def update_usv_full_model(state, n_port, n_stbd, dt, params, _debug=False):
    """
    Otter USV 3-DOF dynamics (Fossen formulation).

    State: [x, y, psi, u, v, r] — NED positions + body-frame velocities.
    Input: n_port, n_stbd — propeller shaft speeds [rad/s].

    Required params keys:
        m11, m22, m33     — total mass matrix diagonal (kg, kg*m^2)
        x_u, y_v, n_r     — damping force coefficients (already signed)
        yaw_nonlinear     — |r| multiplier for yaw damping
        k_pos, k_neg      — thrust coefficients N/(rad/s)^2
        y_pontoon          — pontoon lateral offset [m]
        n_max, n_min       — propeller speed limits [rad/s]
    Optional:
        u_max, v_max, r_max — velocity safety clamps [m/s, rad/s]
    """
    # ── Unpack state ──
    x, y, psi, u, v, r = map(float, state)

    # ── Unpack params ──
    m11 = float(params['m11'])
    m22 = float(params['m22'])
    m33 = float(params['m33'])
    x_u = float(params['x_u'])
    y_v = float(params['y_v'])
    n_r_coef = float(params['n_r'])
    yaw_nl = float(params.get('yaw_nonlinear', 10.0))
    k_pos = float(params['k_pos'])
    k_neg = float(params['k_neg'])
    y_p = float(params['y_pontoon'])
    n_max = float(params['n_max'])
    n_min = float(params['n_min'])

    # ── Clamp propeller speeds ──
    n_port = _clip(float(n_port), n_min, n_max)
    n_stbd = _clip(float(n_stbd), n_min, n_max)

    # ── Quadratic thrust model ──
    T_port = prop_thrust(n_port, k_pos, k_neg)
    T_stbd = prop_thrust(n_stbd, k_pos, k_neg)

    # ── Propulsion generalized forces (Eq. from model.md) ──
    tau_prop = np.array([
        T_port + T_stbd,
        0.0,
        y_p * (T_port - T_stbd)
    ], dtype=float)

    # ── Damping forces (negative form, already signed) ──
    # tau_damp = [x_u*u, y_v*v, n_r*(1 + yaw_nl*|r|)*r]
    tau_damp = np.array([
        x_u * u,
        y_v * v,
        n_r_coef * (1.0 + yaw_nl * abs(r)) * r
    ], dtype=float)

    # ── Coriolis-centripetal term C(nu)*nu ──
    C_nu = np.array([
        -m22 * v * r,
         m11 * u * r,
         (m22 - m11) * u * v
    ], dtype=float)

    # ── Solve M * nu_dot = tau_prop + tau_damp - C_nu ──
    rhs = tau_prop + tau_damp - C_nu
    M_diag = np.array([m11, m22, m33], dtype=float)
    nu_dot = rhs / M_diag

    # ── Semi-implicit Euler: velocity first ──
    nu_new = np.array([u, v, r], dtype=float) + dt * nu_dot

    # ── Velocity safety clamps ──
    u_max_p = float(params.get('u_max', 10.0))
    v_max_p = float(params.get('v_max', 5.0))
    r_max_p = float(params.get('r_max', 3.0))
    nu_new[0] = _clip(nu_new[0], -u_max_p, u_max_p)
    nu_new[1] = _clip(nu_new[1], -v_max_p, v_max_p)
    nu_new[2] = _clip(nu_new[2], -r_max_p, r_max_p)

    u_new, v_new, r_new = nu_new

    # ── Position update using new velocities (semi-implicit) ──
    c, s = np.cos(psi), np.sin(psi)
    x_dot = u_new * c - v_new * s
    y_dot = u_new * s + v_new * c
    psi_dot = r_new

    x_new = x + dt * x_dot
    y_new = y + dt * y_dot
    psi_new = _wrap_angle(psi + dt * psi_dot)

    next_state = np.array([x_new, y_new, psi_new, u_new, v_new, r_new], dtype=float)

    if np.any(np.isnan(next_state)):
        if _debug:
            print("NaN detected in state update")
        return np.zeros(6)

    return next_state
