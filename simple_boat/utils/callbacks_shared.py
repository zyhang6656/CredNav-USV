from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Dict, Any
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class SharedMetricsConfig:
    # --- logging cadence ---
    log_every_steps: int = 4096

    # --- credibility key (optional) ---
    t_key: str = "t_min"
    t_fallback: float = 1.0
    t_low_threshold: float = 0.3

    # --- termination reason keys ---
    reason_key: str = "reason"
    success_reason: str = "goal_reached"
    collision_reasons: Set[str] = None  # if None -> default set below

    # --- optional risk keys present in info ---
    risk_keys: Sequence[str] = ("TCR_sum", "TCR_attn", "num_vo_cones")


class SharedMetricsCallback(BaseCallback):
    """
    A shared callback for BOTH:
      - baseline SB3 PPO (does NOT use t_min in policy)
      - CWVL / Credibility-NLL variants

    It logs from `infos` only. `t_min` is optional:
      - if present: log stats
      - if absent: fallback to t_fallback and track missing fraction
    """

    def __init__(self, cfg: Optional[SharedMetricsConfig] = None, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg or SharedMetricsConfig()
        if self.cfg.collision_reasons is None:
            self.cfg.collision_reasons = {"dynamic_obs", "static_obs", "out_of_bounds", "collision"}

        self.num_envs: int = 1
        self._reset_global()
        self._reset_per_env()

    def _reset_global(self) -> None:
        # step-level aggregates over the last logging window
        self._n_steps = 0
        self._t_sum = 0.0
        self._t_min = 1.0
        self._t_low_cnt = 0
        self._t_missing_cnt = 0

        self._succ_done_cnt = 0
        self._coll_done_cnt = 0
        self._done_cnt = 0

        self._risk_sum = {k: 0.0 for k in self.cfg.risk_keys}
        self._risk_cnt = {k: 0 for k in self.cfg.risk_keys}

    def _reset_per_env(self) -> None:
        # episode accumulators per env index
        self._ep_t_sum = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_t_min = np.ones(self.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self._ep_t_missing = np.zeros(self.num_envs, dtype=np.int64)

    def _on_training_start(self) -> None:
        # stable-baselines3 vec env
        try:
            self.num_envs = int(self.training_env.num_envs)
        except Exception:
            self.num_envs = 1
        self._reset_global()
        self._reset_per_env()

    @staticmethod
    def _to_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        # handle numpy scalar / 0-d array
        try:
            arr = np.asarray(x)
            if arr.shape == ():
                return float(arr)
        except Exception:
            pass
        return None

    def _get_t(self, info: Dict[str, Any]) -> (float, bool):
        """
        Returns (t_value, missing_flag)
        """
        if info is None:
            return self.cfg.t_fallback, True
        if self.cfg.t_key in info:
            t = self._to_float(info.get(self.cfg.t_key))
            if t is not None and np.isfinite(t):
                # clamp to [0, 1] for safety (optional)
                t = float(np.clip(t, 0.0, 1.0))
                return t, False
        return self.cfg.t_fallback, True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None:
            return True

        if dones is None:
            # some training loops may not provide dones; treat as all False
            dones = [False] * len(infos)

        # align lengths robustly
        n = min(len(infos), len(dones))
        for i in range(n):
            info = infos[i] if isinstance(infos[i], dict) else {}
            done = bool(dones[i])

            # --- credibility stats (step-level + per-episode) ---
            t, missing = self._get_t(info)
            self._n_steps += 1
            self._t_sum += t
            self._t_min = min(self._t_min, t)
            if t < self.cfg.t_low_threshold:
                self._t_low_cnt += 1
            if missing:
                self._t_missing_cnt += 1

            if i < self.num_envs:
                self._ep_len[i] += 1
                self._ep_t_sum[i] += t
                self._ep_t_min[i] = min(self._ep_t_min[i], t)
                if missing:
                    self._ep_t_missing[i] += 1

            # --- risk keys (optional) ---
            for k in self.cfg.risk_keys:
                if k in info:
                    v = self._to_float(info.get(k))
                    if v is not None and np.isfinite(v):
                        self._risk_sum[k] += v
                        self._risk_cnt[k] += 1

            # --- episode termination reason (only count on done) ---
            if done:
                self._done_cnt += 1
                r = str(info.get(self.cfg.reason_key, ""))

                if r == self.cfg.success_reason:
                    self._succ_done_cnt += 1
                elif r in self.cfg.collision_reasons:
                    self._coll_done_cnt += 1

                # log per-episode t stats (one scalar per finished episode)
                if i < self.num_envs and self._ep_len[i] > 0:
                    ep_t_mean = float(self._ep_t_sum[i] / max(1, self._ep_len[i]))
                    ep_t_min = float(self._ep_t_min[i])
                    ep_t_missing_frac = float(self._ep_t_missing[i] / max(1, self._ep_len[i]))

                    self.logger.record("ep/t_min", ep_t_min)
                    self.logger.record("ep/t_mean", ep_t_mean)
                    self.logger.record("ep/t_missing_frac", ep_t_missing_frac)

                    # reset per-env episode accumulators
                    self._ep_t_sum[i] = 0.0
                    self._ep_t_min[i] = 1.0
                    self._ep_len[i] = 0
                    self._ep_t_missing[i] = 0

        # --- periodic dump ---
        if (self.num_timesteps % int(self.cfg.log_every_steps)) == 0 and self._n_steps > 0:
            self.logger.record("env/t_mean", float(self._t_sum / self._n_steps))
            self.logger.record("env/t_min", float(self._t_min))
            self.logger.record("env/t_low_frac", float(self._t_low_cnt / self._n_steps))
            self.logger.record("env/t_missing_frac", float(self._t_missing_cnt / self._n_steps))

            # termination outcome rate in the last window
            if self._done_cnt > 0:
                self.logger.record("env/success_rate_window", float(self._succ_done_cnt / self._done_cnt))
                self.logger.record("env/collision_rate_window", float(self._coll_done_cnt / self._done_cnt))

            for k in self.cfg.risk_keys:
                if self._risk_cnt[k] > 0:
                    self.logger.record(f"env/{k}_mean", float(self._risk_sum[k] / self._risk_cnt[k]))

            # self.logger.dump(self.num_timesteps)
            self._reset_global()

        return True
