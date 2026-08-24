"""Innovation-consistency trust factor for CW-VL.

Computes per-obstacle trust based on filter innovation consistency:
  nu_k = z_k - H * x_hat_{k|k-1}
  S_k   = H * P_{k|k-1} * H^T + R_nominal
  S_emp = Cov(nu_{k-W:k})
  d_k   = ||S_emp - S_k||_F / (||S_k||_F + eps)
  t_k   = 1 / (1 + d_k)

Multi-target aggregation: min over risk-active targets.
"""

import numpy as np
from typing import List, Tuple


def _is_risk_active(
    dx: float, dy: float,
    vx_obs: float, vy_obs: float,
    u_own: float, v_own: float,
    risk_distance_threshold: float,
    tcpa_horizon: float,
) -> bool:
    """Check if an obstacle is risk-active based on distance and TCPA.

    Args:
        dx, dy: Relative position (body frame).
        vx_obs, vy_obs: Obstacle velocity (body frame).
        u_own, v_own: Own ship velocity (body frame).
        risk_distance_threshold: Max distance for risk (m).
        tcpa_horizon: TCPA window (s).

    Returns:
        True if obstacle is close AND has positive TCPA within horizon.
    """
    dist = float(np.hypot(dx, dy))
    if dist >= risk_distance_threshold:
        return False

    vx_rel = vx_obs - u_own
    vy_rel = vy_obs - v_own
    v_rel_sq = vx_rel * vx_rel + vy_rel * vy_rel

    if v_rel_sq < 1e-6:
        # Relative stationary: close but not closing => risk-active
        return dist < risk_distance_threshold * 0.5

    tcpa = -(dx * vx_rel + dy * vy_rel) / v_rel_sq
    return 0.0 < tcpa < tcpa_horizon


class InnovationConsistencyTrust:
    """Sliding-window innovation-consistency trust factor.

    Maintains per-obstacle sliding windows of innovations and computes
    trust as the inverse of the discrepancy between empirical and
    filter-predicted innovation covariance.
    """

    def __init__(
        self,
        window_size: int = 20,
        t_min: float = 0.05,
        t_max: float = 1.0,
        risk_distance_threshold: float = 8.0,
        tcpa_horizon: float = 10.0,
        aggregate: str = "min_over_risk_active",
        innovation_mode: str = "position_only_2d",
    ):
        self.window_size = window_size
        self.t_min = t_min
        self.t_max = t_max
        self.risk_distance_threshold = risk_distance_threshold
        self.tcpa_horizon = tcpa_horizon
        self.aggregate = aggregate
        self.innovation_mode = innovation_mode

        # Per-obstacle innovation buffers: {obs_id: [nu_0, nu_1, ...]}
        self._innovation_buffers: dict[int, list[np.ndarray]] = {}
        # Per-obstacle S_k buffers (for averaging across window)
        self._S_buffers: dict[int, list[np.ndarray]] = {}

    def reset(self) -> None:
        """Clear all buffers (call at episode start)."""
        self._innovation_buffers.clear()
        self._S_buffers.clear()

    def update(
        self,
        obs_id: int,
        innovation: np.ndarray,  # nu_k, shape (4,) in world frame [x, y, vx, vy]
        S_k: np.ndarray,         # innovation covariance, shape (4, 4)
    ) -> float:
        """Update sliding window for one obstacle, return per-obstacle trust.

        Args:
            obs_id: Obstacle identifier.
            innovation: Current innovation vector (measurement - predicted).
            S_k: Filter-predicted innovation covariance at this step.

        Returns:
            Per-obstacle trust t_i in [t_min, t_max].
        """
        nu = np.asarray(innovation, dtype=float).ravel()
        S = np.asarray(S_k, dtype=float)

        if obs_id not in self._innovation_buffers:
            self._innovation_buffers[obs_id] = []
            self._S_buffers[obs_id] = []

        buf_nu = self._innovation_buffers[obs_id]
        buf_S = self._S_buffers[obs_id]

        buf_nu.append(nu.copy())
        buf_S.append(S.copy())

        # Trim to window
        if len(buf_nu) > self.window_size:
            buf_nu.pop(0)
            buf_S.pop(0)

        if len(buf_nu) < 4:
            return self.t_max

        # Project innovation and S_k based on mode
        nu_arr_raw = np.array(buf_nu)  # (W, 4) or (W, 2)
        S_arr_raw = np.array(buf_S)    # (W, 4, 4) or (W, 2, 2)

        if self.innovation_mode == "position_only_2d":
            # Use only [x, y] — extract pos indices
            nu_arr = nu_arr_raw[:, :2]          # (W, 2)
            S_arr = S_arr_raw[:, :2, :2]        # (W, 2, 2)
        elif self.innovation_mode == "normalized_full_state_4d":
            # Whiten by mean S_pred: nu_tilde = L^{-1} nu, compare to I
            # For each step separately (S_pred changes over time)
            nu_tilde_list = []
            for t in range(len(buf_nu)):
                S_t = S_arr_raw[t]  # (4, 4)
                S_t = 0.5 * (S_t + S_t.T) + 1e-9 * np.eye(4)
                try:
                    L = np.linalg.cholesky(S_t)
                    nu_w = np.linalg.solve(L, buf_nu[t])
                    nu_tilde_list.append(nu_w)
                except np.linalg.LinAlgError:
                    nu_tilde_list.append(buf_nu[t])
            nu_arr = np.array(nu_tilde_list)  # (W, 4)
            S_arr = np.tile(np.eye(4)[None, :, :], (len(buf_nu), 1, 1))  # (W, 4, 4) — all identity
        else:  # full_state_4d
            nu_arr = nu_arr_raw
            S_arr = S_arr_raw

        # Covariance dimension
        d = nu_arr.shape[1]

        if len(buf_nu) < d + 2:
            return self.t_max

        S_emp = np.cov(nu_arr, rowvar=False)  # (d, d)
        S_mean = np.mean(S_arr, axis=0)       # (d, d)

        norm_S = float(np.linalg.norm(S_mean, ord="fro"))
        if norm_S < 1e-12:
            return self.t_max

        discrepancy = float(np.linalg.norm(S_emp - S_mean, ord="fro")) / norm_S
        t_raw = 1.0 / (1.0 + discrepancy)
        return float(np.clip(t_raw, self.t_min, self.t_max))

    def get_global_trust(
        self,
        obstacle_estimates: dict[int, dict],
        u_own: float,
        v_own: float,
    ) -> float:
        """Aggregate per-obstacle trust into global trust.

        Only risk-active obstacles (close + positive TCPA) are considered.
        Uses min aggregation: t_global = min(t_i for risk-active targets).
        If no risk-active targets, returns 1.0.

        Args:
            obstacle_estimates: {obs_id: {dx, dy, vx, vy, conf, ...}}.
            u_own, v_own: Own ship velocity (body frame).

        Returns:
            Global trust t_global in [t_min, 1.0].
        """
        risk_active_trusts: list[float] = []
        for obs_id, est in obstacle_estimates.items():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            vx_obs = float(est.get("vx", 0.0))
            vy_obs = float(est.get("vy", 0.0))

            if _is_risk_active(
                dx, dy, vx_obs, vy_obs, u_own, v_own,
                self.risk_distance_threshold, self.tcpa_horizon,
            ):
                # Use stored per-obstacle trust, or compute from latest
                t_i = float(est.get("trust", self.t_max))
                risk_active_trusts.append(t_i)

        if not risk_active_trusts:
            return self.t_max

        if self.aggregate == "min_over_risk_active":
            return float(np.clip(min(risk_active_trusts), self.t_min, self.t_max))
        # fallback: mean
        return float(np.clip(np.mean(risk_active_trusts), self.t_min, self.t_max))

    def compute_innovations_batch(
        self,
        z: np.ndarray,      # measurement, shape (4,)
        x_pred: np.ndarray, # predicted state, shape (4,)
        H: np.ndarray,      # measurement matrix (4, 4)
        P_pred: np.ndarray, # predicted covariance (4, 4)
        R_filter: np.ndarray, # filter measurement noise covariance (4, 4)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute innovation and innovation covariance.

        Args:
            z: Measurement vector [x, y, vx, vy].
            x_pred: Predicted state.
            H: Measurement matrix.
            P_pred: Predicted covariance.
            R_filter: Filter measurement noise covariance.

        Returns:
            (innovation nu, innovation_covariance S)
        """
        nu = z - H @ x_pred
        S = H @ P_pred @ H.T + R_filter
        S = 0.5 * (S + S.T)  # symmetrize
        return nu, S


class OracleTrust:
    """Oracle trust using true state (available in simulation only).

    Computes per-target trust from the discrepancy between filter-calculated
    covariance P_f and the true estimation error covariance P_m = e * e^T.

    t_k = 1 / (||P_f - P_m||_2 + 1)

    where ||.||_2 is the matrix spectral norm (not Frobenius).

    This is NOT deployable — requires true state.
    Used ONLY to validate the CW-VL critic weighting mechanism.
    """

    def __init__(
        self,
        t_min: float = 0.05,
        t_max: float = 1.0,
        risk_distance_threshold: float = 10.0,
        tcpa_horizon: float = 12.0,
        aggregate: str = "min_over_risk_active",
        norm_type: str = "spectral_2",
    ):
        self.t_min = t_min
        self.t_max = t_max
        self.risk_distance_threshold = risk_distance_threshold
        self.tcpa_horizon = tcpa_horizon
        self.aggregate = aggregate
        self.norm_type = norm_type  # "spectral_2" or "frobenius"

    def reset(self) -> None:
        """No-op: oracle trust has no sliding window to reset."""
        pass

    def compute_per_target(
        self,
        x_true: np.ndarray,   # true state [x, y, vx, vy]
        x_hat: np.ndarray,    # KF posterior estimate [x, y, vx, vy]
        P_f: np.ndarray,      # KF posterior covariance (4, 4)
    ) -> float:
        """Compute per-target oracle trust from true estimation error.

        P_m = e * e^T where e = x_true - x_hat
        t   = 1 / (||P_f - P_m|| + 1)
        """
        e = (np.asarray(x_true, dtype=float).ravel()
             - np.asarray(x_hat, dtype=float).ravel()).reshape(-1, 1)
        P_m = e @ e.T  # (4, 4)
        diff = np.asarray(P_f, dtype=float) - P_m

        if self.norm_type == "spectral_2":
            # Spectral norm (largest singular value)
            d = float(np.linalg.norm(diff, ord=2))
        else:
            d = float(np.linalg.norm(diff, ord="fro"))

        t_raw = 1.0 / (d + 1.0)
        return float(np.clip(t_raw, self.t_min, self.t_max))

    def get_global_trust(
        self,
        obstacle_estimates: dict[int, dict],
        u_own: float,
        v_own: float,
    ) -> float:
        """Aggregate per-target oracle trust: min over risk-active, else 1.0."""
        risk_active_trusts: list[float] = []
        for obs_id, est in obstacle_estimates.items():
            dx = float(est.get("dx", 0.0))
            dy = float(est.get("dy", 0.0))
            vx_obs = float(est.get("vx", 0.0))
            vy_obs = float(est.get("vy", 0.0))
            if _is_risk_active(
                dx, dy, vx_obs, vy_obs, u_own, v_own,
                self.risk_distance_threshold, self.tcpa_horizon,
            ):
                t_i = float(est.get("trust", self.t_max))
                risk_active_trusts.append(t_i)
        if not risk_active_trusts:
            return self.t_max
        return float(np.clip(min(risk_active_trusts), self.t_min, self.t_max))
