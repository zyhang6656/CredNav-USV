"""Burst mismatch and bias noise injection for training contamination.

Provides configurable burst/bias noise that creates intermittent
filter-model mismatch. The true measurement noise is amplified while
the filter continues to use nominal R, creating contaminated state estimates.

Burst start modes:
  - independent_random: burst window placed randomly in episode (time-based)
  - risk_onset: burst triggers at FIRST risk_active event, then lasts duration
  - path_progress_random_candidate: choose one proxy path-risk step per episode
  - path_progress_binned_candidate: choose one raw proxy path-risk step, then quantize to a binned anchor
  - path_progress_multi_binned_candidate: use several coarse path-risk anchors per episode
  - path_progress_topk_binned_candidate: use local nearest-risk anchors per episode
  - mixed: 50% independent_random, 50% risk_onset per burst episode
"""

import numpy as np


class BurstMismatchInjector:
    """Injects burst measurement noise and position bias into KF measurements.

    Burst: Random episodes have high-noise windows.
      During burst, true measurement covariance = measurement_cov_scale * R_nominal.
      Filter still uses R_nominal (mismatch).

    Bias: Optional lateral position bias added to obstacle measurements.
    """

    def __init__(
        self,
        burst_enabled: bool = True,
        burst_episode_prob: float = 0.6,
        burst_duration_steps: int = 60,
        measurement_cov_scale: float = 100.0,
        bias_enabled: bool = True,
        bias_position_y: float = 0.75,
        bias_duration_steps: int = 30,
        start_mode: str = "mixed",
        trigger_on_risk_active: bool = True,  # deprecated, use start_mode
        measurement_delay_steps: int = 0,
        risk_distance_threshold: float = 10.0,
        tcpa_horizon: float = 12.0,
        nominal_position_std: float = 0.10,
        nominal_velocity_std: float = 0.03,
    ):
        self.measurement_delay_steps = measurement_delay_steps
        self.burst_enabled = burst_enabled
        self.burst_episode_prob = burst_episode_prob
        self.burst_duration_steps = burst_duration_steps
        self.measurement_cov_scale = measurement_cov_scale
        self.bias_enabled = bias_enabled
        self.bias_position_y = bias_position_y
        self.bias_duration_steps = bias_duration_steps
        self.start_mode = start_mode
        self.trigger_on_risk_active = trigger_on_risk_active
        self.risk_distance_threshold = risk_distance_threshold
        self.tcpa_horizon = tcpa_horizon
        self.nominal_position_std = nominal_position_std
        self.nominal_velocity_std = nominal_velocity_std

        # Episode state
        self._burst_allowed = False
        self._burst_armed = False     # risk_onset: waiting for first risk_active
        self._burst_start_step: int = -1
        self._burst_start_steps: list[int] = []
        self._burst_mode_used: str = "none"
        self._bias_active_this_episode = False
        self._bias_start_step: int = -1

        # Stats
        self.total_steps = 0
        self.burst_steps = 0
        self.low_trust_steps = 0
        self.risk_active_steps = 0
        self.burst_and_risk_active_steps = 0

    def reset(self, rng: np.random.Generator, T: int,
              effective_horizon: int = None) -> None:
        """Reset per-episode state.

        Args:
            rng: Seeded RNG.
            T: Full trajectory length (scenario_T).
            effective_horizon: Actual episode horizon for burst placement
                (min(max_episode_steps, scenario_T)). Burst never placed beyond this.
        """
        if effective_horizon is None:
            effective_horizon = T
        eff_h = int(effective_horizon)

        self._burst_allowed = False
        self._burst_armed = False
        self._burst_start_step = -1
        self._burst_start_steps = []
        self._bias_active_this_episode = False
        self._bias_start_step = -1
        self._burst_mode_used = "none"
        self._burst_truncated = False
        self._burst_after_episode = False
        self._invalid_window = False
        self._effective_horizon = eff_h

        if self.burst_enabled and rng.random() < self.burst_episode_prob:
            self._burst_allowed = True
            mode = self.start_mode

            if mode == "mixed":
                mode = "risk_onset" if rng.random() < 0.5 else "independent_random"

            if mode == "independent_random":
                warmup = int(eff_h * 0.05)
                latest_start = eff_h - self.burst_duration_steps
                if latest_start <= warmup:
                    self._invalid_window = True
                    self._burst_allowed = False
                else:
                    self._burst_start_step = int(rng.integers(warmup, latest_start + 1))
                    self._burst_mode_used = "independent_random"
            elif mode in (
                "risk_onset",
                "path_progress",
                "path_progress_random_candidate",
                "path_progress_binned_candidate",
                "path_progress_multi_binned_candidate",
                "path_progress_topk_binned_candidate",
            ):
                self._burst_armed = True
                self._burst_mode_used = mode
            else:
                self._burst_mode_used = mode

        if self.bias_enabled and rng.random() < self.burst_episode_prob:
            self._bias_active_this_episode = True
            warmup = int(eff_h * 0.05)
            latest_start = eff_h - self.bias_duration_steps
            if latest_start > warmup:
                self._bias_start_step = int(rng.integers(warmup, latest_start + 1))

    def set_burst_start_steps(self, starts: list[int] | tuple[int, ...]) -> None:
        """Configure several deterministic burst windows for this episode."""
        clean = sorted(set(int(s) for s in starts if int(s) >= 0))
        self._burst_start_steps = clean
        self._burst_start_step = clean[0] if clean else -1
        self._burst_armed = False
        self._burst_allowed = bool(clean)

    def is_burst(self, step: int, is_risk_active: bool) -> bool:
        """Check if burst is active at this step."""
        if not self._burst_allowed:
            return False

        if self._burst_start_steps:
            return any(
                int(start) <= step < int(start) + self.burst_duration_steps
                for start in self._burst_start_steps
            )

        # risk_onset: arm and wait for first risk_active (within effective horizon)
        if self._burst_armed:
            if is_risk_active and step < self._effective_horizon:
                self._burst_start_step = step
                self._burst_armed = False
                # Truncation: burst ends at min(start+duration, effective_horizon)
                end = self._burst_start_step + self.burst_duration_steps
                if end > self._effective_horizon:
                    self._burst_truncated = True
            else:
                return False

        if self._burst_start_step < 0:
            # Check: burst was expected but never triggered
            if step >= self._effective_horizon:
                self._burst_after_episode = True
            return False

        return self._burst_start_step <= step < self._burst_start_step + self.burst_duration_steps

    @property
    def burst_triggered(self) -> bool:
        """Whether burst was actually triggered this episode (fired at least once)."""
        return self._burst_allowed and (self._burst_start_step >= 0 or bool(self._burst_start_steps))

    def is_bias(self, step: int) -> bool:
        if not self._bias_active_this_episode:
            return False
        return self._bias_start_step <= step < self._bias_start_step + self.bias_duration_steps

    def get_true_measurement_covariance(self, step: int, is_risk_active: bool) -> np.ndarray:
        R_nom = np.diag([
            self.nominal_position_std ** 2,
            self.nominal_position_std ** 2,
            self.nominal_velocity_std ** 2,
            self.nominal_velocity_std ** 2,
        ]).astype(float)

        if self.is_burst(step, is_risk_active):
            return R_nom * float(self.measurement_cov_scale)

        return R_nom

    def get_bias(self, step: int) -> float:
        if self.is_bias(step):
            return float(self.bias_position_y)
        return 0.0

    def update_stats(self, step: int, is_risk_active: bool, trust: float,
                     trust_threshold: float = 0.3) -> None:
        self.total_steps += 1
        burst_now = self.is_burst(step, is_risk_active)
        if burst_now:
            self.burst_steps += 1
        if trust < trust_threshold:
            self.low_trust_steps += 1
        if is_risk_active:
            self.risk_active_steps += 1
        if burst_now and is_risk_active:
            self.burst_and_risk_active_steps += 1

    def get_ratios(self) -> dict:
        denom = max(self.total_steps, 1)
        return {
            "low_trust_step_ratio": self.low_trust_steps / denom,
            "burst_step_ratio": self.burst_steps / denom,
            "risk_active_step_ratio": self.risk_active_steps / denom,
            "burst_and_risk_active_step_ratio": self.burst_and_risk_active_steps / denom,
        }

    def reset_stats(self) -> None:
        self.total_steps = 0
        self.burst_steps = 0
        self.low_trust_steps = 0
        self.risk_active_steps = 0
        self.burst_and_risk_active_steps = 0
