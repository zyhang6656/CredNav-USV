from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from simple_boat.envs.dqn_vo_baseline import tcpa_dcpa
from simple_boat.envs.dynamics import update_usv_full_model


@dataclass
class COLREGSMPCBaseline:
    horizon_steps: int = 41
    safety_distance: float = 2.0
    warning_distance: float = 6.0
    yaw_rate_scale: float = 0.8
    cruise_action: float = 0.65
    prediction_dt: float = 0.25
    control_period: float = 0.25
    integration_substeps: int = 1
    encounter_radius: float = 21.0
    emergency_radius: float = 10.0
    head_on_course_deg: float = math.degrees(0.25)
    optimizer_maxiter: int = 30
    reference_speed: float = 1.0
    q_contour: float = 0.1
    q_lag: float = 0.001
    q_speed: float = 1.0
    q_speed_em: float = 0.01
    q_lateral_velocity: float = 0.25
    q_control_surge: float = 0.0001
    q_control_yaw: float = 0.003
    colregs_alpha_gw: float = 0.97
    colregs_alpha_em: float = 0.05
    ov_length: float = 2.0
    ov_width: float = 1.08
    geometry_margin_scale: float = 1.0
    _roles: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _last_actions: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_pred_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_s: np.ndarray | None = field(default=None, init=False, repr=False)
    _solver_cache: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _step_fn: Any = field(default=None, init=False, repr=False)
    _last_dyn_step: int = field(default=-1, init=False, repr=False)
    _held_action: np.ndarray | None = field(default=None, init=False, repr=False)
    _held_score: float = field(default=float("inf"), init=False, repr=False)
    _next_solve_time: float = field(default=0.0, init=False, repr=False)
    _last_solver_status: str = field(default="not_solved", init=False, repr=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "COLREGSMPCBaseline":
        cfg = dict(config.get("colregs_mpc", config.get("controller", {})))
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in allowed})

    def predict(self, env: Any) -> tuple[np.ndarray, dict[str, Any]]:
        role_items = self._update_roles(env)
        role, _threat = self._dominant_role_from_items(role_items)
        active_constraints = sum(1 for item in role_items if self._is_active_constraint_item(item))
        if self._should_hold_action(env):
            action = np.clip(np.asarray(self._held_action, dtype=float), -1.0, 1.0).astype(np.float32)
            return action, self._info(role, active_constraints, self._held_score, solver_failed=False, held=True)
        best_action, best_score, solver_failed = self._optimize_action_sequence(env, role_items, role)
        self._held_action = np.clip(best_action, -1.0, 1.0).astype(float)
        self._held_score = float(best_score)
        self._advance_next_solve_time(env)
        return np.clip(best_action, -1.0, 1.0).astype(np.float32), self._info(role, active_constraints, best_score, solver_failed, held=False)

    def _info(self, role: str, active_constraints: int, score: float, solver_failed: bool, held: bool) -> dict[str, Any]:
        return {
            "colregs_role": role,
            "colregs_mpc_cost": float(score),
            "colregs_mpc_path_len": 2,
            "colregs_mpc_reference": "straight",
            "colregs_mpc_active_constraints": int(active_constraints),
            "colregs_mpc_solver": "casadi_ipopt",
            "colregs_mpc_solver_failed": bool(solver_failed),
            "colregs_mpc_solver_status": "held" if held else self._last_solver_status,
            "colregs_mpc_held": bool(held),
        }

    def _sim_time(self, env: Any) -> float:
        return float(getattr(env, "dyn_step", 0)) * float(getattr(env, "dt", 0.1))

    def _should_hold_action(self, env: Any) -> bool:
        if self._held_action is None or float(self.control_period) <= float(getattr(env, "dt", 0.1)):
            return False
        return self._sim_time(env) < self._next_solve_time - 1e-9

    def _advance_next_solve_time(self, env: Any) -> None:
        period = float(self.control_period)
        if period <= 0.0:
            self._next_solve_time = self._sim_time(env)
            return
        now = self._sim_time(env)
        if self._next_solve_time <= 0.0:
            self._next_solve_time = now
        while self._next_solve_time <= now + 1e-9:
            self._next_solve_time += period

    def _dominant_role(self, env: Any) -> tuple[str, dict[str, float] | None]:
        return self._dominant_role_from_items(self._update_roles(env))

    def _dominant_role_from_items(self, items: list[dict[str, Any]]) -> tuple[str, dict[str, float] | None]:
        priority = {"EM": 3, "GW": 2, "SO": 1, "NONE": 0}
        best_role = "NONE"
        best: dict[str, float] | None = None
        best_score = 0.0
        for item in items:
            role = str(item["role"])
            score = priority.get(role, 0) + max(0.0, 1.0 - float(item["dcpa"]) / max(self.warning_distance, 1e-6))
            if score > best_score:
                best_score = score
                best_role = role
                best = {k: float(item[k]) for k in ("dx", "dy", "vx", "vy", "tcpa", "dcpa", "dist")}
        return best_role, best

    def _update_roles(self, env: Any) -> list[dict[str, Any]]:
        dyn_step = int(getattr(env, "dyn_step", 0))
        if dyn_step == 0 and self._last_dyn_step > 0:
            self._roles.clear()
            self._last_actions = None
            self._last_pred_positions = None
            self._last_s = None
            self._held_action = None
            self._held_score = float("inf")
            self._next_solve_time = 0.0
            self._last_solver_status = "not_solved"
        self._last_dyn_step = dyn_step

        ship = np.asarray(env.ship_state, dtype=float)
        seen: set[int] = set()
        items: list[dict[str, Any]] = []
        for raw_id, est in getattr(env, "obstacle_estimates", {}).items():
            obs_id = int(raw_id)
            seen.add(obs_id)
            dx, dy = float(est.get("dx", 0.0)), float(est.get("dy", 0.0))
            vx, vy = float(est.get("vx", 0.0)), float(est.get("vy", 0.0))
            tcpa, dcpa = tcpa_dcpa(dx, dy, float(ship[3]), float(ship[4]), vx, vy)
            dist = float(math.hypot(dx, dy))
            prev = self._roles.get(obs_id, "NONE")
            if dist >= self.encounter_radius:
                role = "NONE"
            elif prev == "GW":
                role = "GW"
            elif prev == "EM":
                role = "EM" if dist < self.emergency_radius else "SO"
            elif prev == "SO":
                role = "EM" if dist < self.emergency_radius else "SO"
            else:
                role = self._entry_role(env, est, tcpa, dcpa, dist)
                if role == "SO" and dist < self.emergency_radius:
                    role = "EM"
            if role == "NONE":
                self._roles.pop(obs_id, None)
                continue
            self._roles[obs_id] = role
            items.append({
                "id": obs_id,
                "role": role,
                "dx": dx,
                "dy": dy,
                "vx": vx,
                "vy": vy,
                "tcpa": float(tcpa),
                "dcpa": float(dcpa),
                "dist": dist,
                "radius": float(est.get("radius", getattr(env, "dyn_radius", 1.0))),
            })
        for obs_id in list(self._roles):
            if obs_id not in seen:
                self._roles.pop(obs_id, None)
        return items

    def _entry_role(self, env: Any, est: dict[str, Any], tcpa: float, dcpa: float, dist: float) -> str:
        radius = float(getattr(env, "R_usv", 1.0)) + float(est.get("radius", getattr(env, "dyn_radius", 1.0)))
        risk_limit = radius + max(0.0, float(self.safety_distance))
        if dist >= self.encounter_radius:
            return "NONE"
        if dcpa >= risk_limit:
            return "NONE"
        dx, dy = float(est.get("dx", 0.0)), float(est.get("dy", 0.0))
        psi_beta = math.atan2(-dy, dx) % (2.0 * math.pi)
        psi_course = self._relative_course(est)
        psi_h = math.radians(self.head_on_course_deg)
        head_on = math.pi - psi_h <= psi_course < math.pi + psi_h
        crossing_starboard = math.pi + psi_h <= psi_course < 13.0 * math.pi / 8.0
        bearing_region = psi_course >= 13.0 * math.pi / 8.0 or psi_course < 3.0 * math.pi / 8.0
        overtaking = 5.0 * math.pi / 8.0 <= ((math.pi + psi_beta - psi_course) % (2.0 * math.pi)) < 11.0 * math.pi / 8.0
        starboard_bearing = 0.0 <= psi_beta < 5.0 * math.pi / 8.0
        if head_on or crossing_starboard or (bearing_region and (overtaking or starboard_bearing)):
            return "GW"
        return "SO"

    @staticmethod
    def _relative_course(est: dict[str, Any]) -> float:
        vx, vy = float(est.get("vx", 0.0)), float(est.get("vy", 0.0))
        if math.hypot(vx, vy) <= 1e-6:
            return 0.0
        return math.atan2(-vy, vx) % (2.0 * math.pi)

    def _optimize_action_sequence(
        self,
        env: Any,
        role_items: list[dict[str, Any]],
        role: str,
    ) -> tuple[np.ndarray, float, bool]:
        rows = self._colregs_constraint_rows(env, role_items)
        try:
            data = self._casadi_solver(len(rows))
            z0 = self._initial_decision_guess(env, role=role, rows=rows)
            sol = data["solver"](
                x0=z0,
                lbx=data["lbx"],
                ubx=data["ubx"],
                lbg=data["lbg"],
                ubg=data["ubg"],
                p=self._casadi_parameter_values(env, role_items, rows),
            )
            stats = data["solver"].stats()
            self._last_solver_status = str(stats.get("return_status", "unknown"))
            z = np.asarray(sol["x"], dtype=float).reshape(-1)
            if not np.all(np.isfinite(z)):
                raise RuntimeError("non-finite NLP solution")
            X, U, S, _V = self._unpack_decision(z)
            solver_failed = not bool(stats.get("success", False))
            actions = U.T.copy()
            values = self._constraint_values_for_actions(env, actions, rows)
            if values.size and float(np.min(values)) < -1e-2:
                raise RuntimeError("NLP solution violates hard COLREGs constraints")
            self._last_actions = np.clip(actions, -1.0, 1.0)
            self._last_pred_positions = X[:2, 1:].T.copy()
            self._last_s = S.copy()
            return self._last_actions[0].astype(np.float32), float(sol["f"]), solver_failed
        except Exception as exc:
            self._last_solver_status = f"exception:{type(exc).__name__}"
            if self._last_actions is not None and self._last_actions.shape == (self.horizon_steps, 2):
                action = np.asarray(self._last_actions[0], dtype=float)
                self._last_actions = np.vstack([self._last_actions[1:], self._last_actions[-1:]])
            else:
                action = np.array([self.cruise_action, 0.0], dtype=float)
                self._last_actions = np.tile(action, (self.horizon_steps, 1))
            self._last_pred_positions = self._rollout_positions(env, self._last_actions)
            return action.astype(np.float32), float("inf"), True

    def _casadi_solver(self, n_rows: int) -> dict[str, Any]:
        N = max(1, int(self.horizon_steps))
        key = (N, int(n_rows))
        cached = self._solver_cache.get(key)
        if cached is not None:
            return cached

        X = ca.MX.sym("X", 6, N + 1)
        U = ca.MX.sym("U", 2, N)
        S = ca.MX.sym("S", N + 1)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), S)

        p = ca.MX.sym("p", 35 + 5 * int(n_rows))
        off = 0
        x_init = p[off:off + 6]
        off += 6
        start = p[off:off + 2]
        off += 2
        tangent = p[off:off + 2]
        off += 2
        normal = p[off:off + 2]
        off += 2
        path_len = p[off]
        off += 1
        s_init = p[off]
        off += 1
        q_speed = p[off]
        off += 1
        bounds = p[off:off + 4]
        off += 4
        dyn = p[off:off + 16]
        off += 16
        row_params = ca.reshape(p[off:off + 5 * int(n_rows)], 5, int(n_rows)) if n_rows else None

        g: list[Any] = []
        lbg: list[float] = []
        ubg: list[float] = []

        def eq(expr: Any) -> None:
            g.append(expr)
            lbg.extend([0.0] * int(expr.numel()))
            ubg.extend([0.0] * int(expr.numel()))

        def ge0(expr: Any) -> None:
            g.append(expr)
            lbg.extend([0.0] * int(expr.numel()))
            ubg.extend([float("inf")] * int(expr.numel()))

        eq(X[:, 0] - x_init)
        eq(S[0] - s_init)
        cost = 0
        dt = dyn[0]
        for k in range(N):
            eq(X[:, k + 1] - self._casadi_step_expr(X[:, k], U[:, k], dyn))
            eq(S[k + 1] - (S[k] + dt * X[3, k]))
            ge0(S[k])
            ge0(X[0, k + 1] - bounds[0])
            ge0(bounds[1] - X[0, k + 1])
            ge0(X[1, k + 1] - bounds[2])
            ge0(bounds[3] - X[1, k + 1])

            pos = X[0:2, k + 1]
            ref = start + tangent * S[k + 1]
            err = pos - ref
            psi = X[2, k + 1]
            c, s = ca.cos(psi), ca.sin(psi)
            lag = ca.dot(err, tangent)
            contour = ca.dot(err, normal)
            cost += float(self.q_contour) * contour * contour
            cost += float(self.q_lag) * lag * lag
            cost += q_speed * (X[3, k + 1] - float(self.reference_speed)) ** 2
            cost += float(self.q_lateral_velocity) * X[4, k + 1] ** 2
            cost += float(self.q_control_surge) * U[0, k] ** 2
            cost += float(self.q_control_yaw) * U[1, k] ** 2

        ge0(S[N])
        for r in range(int(n_rows)):
            k = r % N
            row = row_params[:, r]
            value = (X[0, k + 1] - row[2]) * row[0] + (X[1, k + 1] - row[3]) * row[1] - row[4]
            ge0(value)

        lbx = np.full(int(w.numel()), -np.inf, dtype=float)
        ubx = np.full(int(w.numel()), np.inf, dtype=float)
        u0 = 6 * (N + 1)
        u1 = u0 + 2 * N
        lbx[u0:u1] = -1.0
        ubx[u0:u1] = 1.0
        s0 = u1
        s1 = s0 + N + 1
        lbx[s0:s1] = 0.0

        solver = ca.nlpsol(
            "colregs_mpc",
            "ipopt",
            {"x": w, "f": cost, "g": ca.vertcat(*g), "p": p},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": max(1, int(self.optimizer_maxiter)),
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_iter": 3,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.nlp_scaling_method": "gradient-based",
            },
        )
        data = {
            "solver": solver,
            "lbx": lbx,
            "ubx": ubx,
            "lbg": np.asarray(lbg, dtype=float),
            "ubg": np.asarray(ubg, dtype=float),
        }
        self._solver_cache[key] = data
        return data

    def _casadi_parameter_values(self, env: Any, role_items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> np.ndarray:
        start, _goal, tangent, normal, path_len = self._straight_reference(env)
        state = np.asarray(env.ship_state, dtype=float).reshape(-1)[:6]
        s_init = self._project_path_s(state[:2], start, tangent, path_len)
        role, _ = self._dominant_role_from_items(role_items)
        q_speed = self.q_speed_em if role == "EM" else self.q_speed
        row_values: list[float] = []
        for row in rows:
            normal_row = np.asarray(row["normal"], dtype=float)
            point = np.asarray(row["point"], dtype=float)
            row_values.extend([float(normal_row[0]), float(normal_row[1]), float(point[0]), float(point[1]), float(row["rho"])])
        return np.asarray([
            *state.tolist(),
            *start.tolist(),
            *tangent.tolist(),
            *normal.tolist(),
            float(path_len),
            float(s_init),
            float(q_speed),
            *self._position_bounds(env).tolist(),
            *self._dynamics_param_values(env).tolist(),
            *row_values,
        ], dtype=float)

    def _position_bounds(self, env: Any) -> np.ndarray:
        radius = float(getattr(env, "R_usv", 0.0))
        width = float(getattr(env, "W", 32.0))
        height = float(getattr(env, "H", 32.0))
        return np.asarray([radius, width - radius, radius, height - radius], dtype=float)

    def _prediction_dt(self, env: Any) -> float:
        dt = float(self.prediction_dt)
        return dt if dt > 0.0 else float(getattr(env, "dt", 0.1))

    def _initial_decision_guess(
        self,
        env: Any,
        role: str = "NONE",
        rows: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        N = max(1, int(self.horizon_steps))
        if self._last_actions is not None and self._last_actions.shape == (N, 2):
            actions = np.vstack([self._last_actions[1:], self._last_actions[-1:]])
            values = self._constraint_values_for_actions(env, actions, rows) if role == "GW" and rows else np.empty(0)
            if values.size and float(np.min(values)) < -1e-2:
                actions = np.tile(np.array([self.cruise_action, -0.2], dtype=float), (N, 1))
        else:
            initial_yaw = -0.2 if role == "GW" else 0.0
            actions = np.tile(np.array([self.cruise_action, initial_yaw], dtype=float), (N, 1))
        states = self._rollout_states(env, actions, wrap_heading=False)
        start, _goal, tangent, _normal, path_len = self._straight_reference(env)
        s_guess = np.empty(N + 1, dtype=float)
        s_guess[0] = self._project_path_s(states[0, :2], start, tangent, path_len)
        dt = self._prediction_dt(env)
        for k in range(N):
            s_guess[k + 1] = max(0.0, s_guess[k] + dt * float(states[k, 3]))
        return self._pack_decision(states, actions, s_guess)

    def _pack_decision(self, states: np.ndarray, actions: np.ndarray, s_path: np.ndarray) -> np.ndarray:
        X = np.asarray(states, dtype=float).reshape(self.horizon_steps + 1, 6).T
        U = np.asarray(actions, dtype=float).reshape(self.horizon_steps, 2).T
        return np.concatenate([
            X.reshape(-1, order="F"),
            U.reshape(-1, order="F"),
            np.asarray(s_path, dtype=float).reshape(-1),
        ])

    def _unpack_decision(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = max(1, int(self.horizon_steps))
        z = np.asarray(z, dtype=float).reshape(-1)
        x_end = 6 * (N + 1)
        u_end = x_end + 2 * N
        s_end = u_end + N + 1
        X = z[:x_end].reshape(6, N + 1, order="F")
        U = z[x_end:u_end].reshape(2, N, order="F")
        S = z[u_end:s_end]
        V = X[3, :N].copy()
        return X, U, S, V

    @staticmethod
    def _casadi_clip(x: Any, lo: Any, hi: Any) -> Any:
        return ca.fmin(ca.fmax(x, lo), hi)

    def _casadi_step_expr(self, state: Any, action: Any, dyn: Any) -> Any:
        dt, m11, m22, m33, x_u, y_v, n_r_coef, yaw_nl, k_pos, k_neg, y_p, n_max, n_min, u_max, v_max, r_max = [
            dyn[i] for i in range(16)
        ]
        raw_surge = self._casadi_clip(action[0], -1.0, 1.0)
        raw_yaw = self._casadi_clip(action[1], -1.0, 1.0)
        n_common = ca.if_else(raw_surge >= 0.0, raw_surge * n_max, raw_surge * (-n_min)) * 0.70
        n_diff = raw_yaw * ca.fmin(n_max, -n_min)
        n_port = self._casadi_clip(n_common + n_diff, n_min, n_max)
        n_stbd = self._casadi_clip(n_common - n_diff, n_min, n_max)
        t_port = ca.if_else(n_port >= 0.0, k_pos * n_port * ca.fabs(n_port), k_neg * n_port * ca.fabs(n_port))
        t_stbd = ca.if_else(n_stbd >= 0.0, k_pos * n_stbd * ca.fabs(n_stbd), k_neg * n_stbd * ca.fabs(n_stbd))
        tau_u = t_port + t_stbd
        tau_r = y_p * (t_port - t_stbd)
        step_dt = dt / max(1, int(self.integration_substeps))
        next_state = state
        for _ in range(max(1, int(self.integration_substeps))):
            x, y, psi, u, v, r = [next_state[i] for i in range(6)]
            u_dot = (tau_u + x_u * u + m22 * v * r) / m11
            v_dot = (y_v * v - m11 * u * r) / m22
            r_dot = (tau_r + n_r_coef * (1.0 + yaw_nl * ca.fabs(r)) * r - (m22 - m11) * u * v) / m33
            u_new = self._casadi_clip(u + step_dt * u_dot, -u_max, u_max)
            v_new = self._casadi_clip(v + step_dt * v_dot, -v_max, v_max)
            r_new = self._casadi_clip(r + step_dt * r_dot, -r_max, r_max)
            c, s = ca.cos(psi), ca.sin(psi)
            next_state = ca.vertcat(
                x + step_dt * (u_new * c - v_new * s),
                y + step_dt * (u_new * s + v_new * c),
                psi + step_dt * r_new,
                u_new,
                v_new,
                r_new,
            )
        return next_state

    def _casadi_step_numeric(self, env: Any, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self._step_fn is None:
            state_sym = ca.MX.sym("state", 6)
            action_sym = ca.MX.sym("action", 2)
            dyn_sym = ca.MX.sym("dyn", 16)
            self._step_fn = ca.Function("otter_step", [state_sym, action_sym, dyn_sym], [self._casadi_step_expr(state_sym, action_sym, dyn_sym)])
        out = np.asarray(self._step_fn(np.asarray(state, dtype=float), np.asarray(action, dtype=float), self._dynamics_param_values(env)), dtype=float).reshape(-1)
        out[2] = float(out[2]) % (2.0 * math.pi)
        return out

    def _dynamics_param_values(self, env: Any) -> np.ndarray:
        params = getattr(env, "usv_params")
        return np.asarray([
            self._prediction_dt(env),
            float(params["m11"]),
            float(params["m22"]),
            float(params["m33"]),
            float(params["x_u"]),
            float(params["y_v"]),
            float(params["n_r"]),
            float(params.get("yaw_nonlinear", 10.0)),
            float(params["k_pos"]),
            float(params["k_neg"]),
            float(params["y_pontoon"]),
            float(params["n_max"]),
            float(params["n_min"]),
            float(params.get("u_max", 10.0)),
            float(params.get("v_max", 5.0)),
            float(params.get("r_max", 3.0)),
        ], dtype=float)

    def _rollout_states(self, env: Any, actions: np.ndarray, *, wrap_heading: bool = True) -> np.ndarray:
        state = np.asarray(env.ship_state, dtype=float).copy()
        states = [state.copy()]
        for action in np.asarray(actions, dtype=float):
            state = self._next_state_for_action(env, state, action, wrap_heading=wrap_heading)
            states.append(state.copy())
        return np.asarray(states, dtype=float)

    def _rollout_positions(self, env: Any, actions: np.ndarray) -> np.ndarray:
        return self._rollout_states(env, actions)[1:, :2]

    def _constraint_values_for_actions(self, env: Any, actions: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=float)
        positions = self._rollout_positions(env, actions)
        values = [
            float((positions[int(row["k"])] - np.asarray(row["point"], dtype=float)) @ np.asarray(row["normal"], dtype=float) - float(row["rho"]))
            for row in rows
        ]
        return np.asarray(values, dtype=float)

    def _is_active_constraint_item(self, item: dict[str, Any]) -> bool:
        return str(item.get("role", "NONE")) in {"GW", "EM"}

    def _colregs_constraint_rows(self, env: Any, role_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active = [item for item in role_items if self._is_active_constraint_item(item)]
        if not active:
            return []
        anchors = self._constraint_anchors(env)
        obs_world = self._obstacles_world_dicts(env, active)
        dt = self._prediction_dt(env)
        rows: list[dict[str, Any]] = []
        for obs in obs_world:
            for k, anchor in enumerate(anchors):
                center = np.array([
                    float(obs["x"] + obs["vx"] * (k + 1) * dt),
                    float(obs["y"] + obs["vy"] * (k + 1) * dt),
                ], dtype=float)
                heading = math.atan2(float(obs["vy"]), float(obs["vx"])) if math.hypot(float(obs["vx"]), float(obs["vy"])) > 1e-6 else 0.0
                rho = float(getattr(env, "R_usv", 1.0))
                scale = max(0.0, float(self.geometry_margin_scale))
                margins = (
                    scale * float(self.ov_length),
                    scale * 0.5 * float(self.ov_length),
                    scale * float(self.ov_width),
                    scale * float(self.ov_width),
                )
                rows.append(self._constraint_row(k, anchor, center, heading, str(obs["role"]), rho, margins=margins))
        return rows

    def _constraint_anchors(self, env: Any) -> np.ndarray:
        N = max(1, int(self.horizon_steps))
        if self._last_pred_positions is not None and self._last_pred_positions.shape == (N, 2):
            last = np.asarray(self._last_pred_positions, dtype=float)
            tail = 2.0 * last[-1] - last[-2] if len(last) > 1 else last[-1]
            return np.vstack([last[1:], tail])
        return self._rollout_positions(env, np.tile(np.array([self.cruise_action, 0.0], dtype=float), (N, 1)))

    def _constraint_row(
        self,
        k: int,
        anchor: np.ndarray,
        center: np.ndarray,
        heading: float,
        role: str,
        rho: float,
        margin: float = 0.0,
        margins: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        vertices = self._rectangle_vertices(center, heading, margin=margin, margins=margins)
        anchor = np.asarray(anchor, dtype=float)
        target_alpha = self.colregs_alpha_gw if role == "GW" else self.colregs_alpha_em

        def valid_candidates(alpha: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
            out: list[tuple[float, np.ndarray, np.ndarray]] = []
            for vertex in vertices:
                normal = anchor - vertex
                norm = float(np.linalg.norm(normal))
                if norm < 1e-9:
                    continue
                normal = normal / norm
                theta = math.acos(float(np.clip(float(rho) / norm, -1.0, 1.0))) if norm > float(rho) else 0.0
                angle = float(alpha) * theta
                c, s = math.cos(angle), math.sin(angle)
                rotated = np.array([c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]], dtype=float)
                if max(float(rotated @ (other - vertex)) for other in vertices) <= 1e-9:
                    out.append((float(rotated @ (anchor - vertex) - float(rho)), rotated, vertex))
            return out

        candidates = valid_candidates(float(target_alpha))
        if not candidates and float(target_alpha) > 0.0:
            low, high = 0.0, float(target_alpha)
            candidates = valid_candidates(low)
            for _ in range(20):
                mid = 0.5 * (low + high)
                trial = valid_candidates(mid)
                if trial:
                    low, candidates = mid, trial
                else:
                    high = mid
        if candidates:
            feasible = [item for item in candidates if item[0] >= -1e-9]
            margin_value, rotated, vertex = min(feasible, key=lambda item: item[0]) if feasible else max(candidates, key=lambda item: item[0])
            _ = margin_value
        else:
            rotated = anchor - np.asarray(center, dtype=float)
            norm = float(np.linalg.norm(rotated))
            rotated = rotated / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=float)
            vertex = max(vertices, key=lambda item: float(rotated @ item))
        return {"k": int(k), "normal": rotated, "point": vertex, "rho": float(rho)}

    def _rectangle_vertices(
        self,
        center: np.ndarray,
        heading: float,
        margin: float = 0.0,
        margins: tuple[float, float, float, float] | None = None,
    ) -> list[np.ndarray]:
        c, s = math.cos(float(heading)), math.sin(float(heading))
        rot = np.array([[c, -s], [s, c]], dtype=float)
        if margins is None:
            extra = max(0.0, float(margin))
            bow = stern = 0.5 * float(self.ov_length) + extra
            port = starboard = 0.5 * float(self.ov_width) + extra
        else:
            bm, sn, pt, sb = (max(0.0, float(value)) for value in margins)
            bow = 0.5 * float(self.ov_length) + bm
            stern = 0.5 * float(self.ov_length) + sn
            port = 0.5 * float(self.ov_width) + pt
            starboard = 0.5 * float(self.ov_width) + sb
        local_vertices = (
            np.array([bow, port], dtype=float),
            np.array([bow, -starboard], dtype=float),
            np.array([-stern, -starboard], dtype=float),
            np.array([-stern, port], dtype=float),
        )
        return [np.asarray(center, dtype=float) + rot @ vertex for vertex in local_vertices]

    def _straight_reference(self, env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        ship = np.asarray(env.ship_state, dtype=float)
        start = np.asarray(getattr(env, "initial_position", ship[:2]), dtype=float).reshape(-1)[:2]
        goal = np.asarray(env.goal, dtype=float).reshape(-1)[:2]
        delta = goal - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            tangent = np.array([math.cos(float(ship[2])), math.sin(float(ship[2]))], dtype=float)
            length = 1.0
        else:
            tangent = delta / length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        return start, goal, tangent, normal, length

    @staticmethod
    def _project_path_s(pos: np.ndarray, start: np.ndarray, tangent: np.ndarray, path_len: float) -> float:
        return max(0.0, float((np.asarray(pos, dtype=float) - start) @ tangent))

    def _next_state_for_action(self, env: Any, state: np.ndarray, action: np.ndarray, *, wrap_heading: bool = True) -> np.ndarray:
        if hasattr(env, "_action_to_prop_speeds"):
            n_port, n_stbd = env._action_to_prop_speeds(np.asarray(action, dtype=float))
        else:
            raw_surge = float(np.clip(action[0], -1.0, 1.0))
            raw_yaw = float(np.clip(action[1], -1.0, 1.0))
            params = getattr(env, "usv_params", {})
            n_max = float(params.get("n_max", 103.930864274))
            n_min = float(params.get("n_min", -101.736665504))
            n_common = (raw_surge * n_max if raw_surge >= 0.0 else raw_surge * abs(n_min)) * 0.70
            n_diff = raw_yaw * min(n_max, abs(n_min))
            n_port = float(np.clip(n_common + n_diff, n_min, n_max))
            n_stbd = float(np.clip(n_common - n_diff, n_min, n_max))
        substeps = max(1, int(self.integration_substeps))
        dt = self._prediction_dt(env) / substeps
        next_state = np.asarray(state, dtype=float).copy()
        for _ in range(substeps):
            previous = next_state
            next_state = update_usv_full_model(
                previous,
                float(n_port),
                float(n_stbd),
                dt,
                getattr(env, "usv_params"),
            )
            if not wrap_heading:
                next_state[2] = float(previous[2]) + dt * float(next_state[5])
        if wrap_heading:
            next_state[2] = float(next_state[2]) % (2.0 * math.pi)
        return next_state

    def _obstacles_world(self, env: Any) -> list[tuple[float, float, float, float, float]]:
        return [
            (float(obs["x"]), float(obs["y"]), float(obs["vx"]), float(obs["vy"]), float(obs["radius"]))
            for obs in self._obstacles_world_dicts(env)
        ]

    def _obstacles_world_dicts(
        self,
        env: Any,
        role_items: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ship = np.asarray(env.ship_state, dtype=float)
        c, s = math.cos(float(ship[2])), math.sin(float(ship[2]))
        out = []
        if role_items is None:
            role_items = [
                {
                    "id": int(obs_id),
                    "role": "NONE",
                    "dx": float(est.get("dx", 0.0)),
                    "dy": float(est.get("dy", 0.0)),
                    "vx": float(est.get("vx", 0.0)),
                    "vy": float(est.get("vy", 0.0)),
                    "radius": float(est.get("radius", getattr(env, "dyn_radius", 1.0))),
                }
                for obs_id, est in getattr(env, "obstacle_estimates", {}).items()
            ]
        for item in role_items:
            dx, dy = float(item.get("dx", 0.0)), float(item.get("dy", 0.0))
            vx, vy = float(item.get("vx", 0.0)), float(item.get("vy", 0.0))
            out.append({
                "id": int(item.get("id", len(out))),
                "role": str(item.get("role", "NONE")),
                "x": float(ship[0] + c * dx - s * dy),
                "y": float(ship[1] + s * dx + c * dy),
                "vx": float(c * vx - s * vy),
                "vy": float(s * vx + c * vy),
                "radius": float(item.get("radius", getattr(env, "dyn_radius", 1.0))),
            })
        return out
