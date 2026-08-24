"""Unified actor-critic policy for PPO, Hetero-PPO, and credibility-weighted PPO.

PPO baseline (value_loss_type="mse"):
  - Critic outputs single value V_phi(b).
  - Actor uses standard PPO-Clip.

Credibility-weighted PPO (value_loss_type="cwvl"):
  - Critic outputs mu_phi(b) and s_phi(b) = log(sigma_phi^2).
  - log_var is trainable (freeze_sigma=False) and clamped.

Hetero-PPO (value_loss_type="hetero") uses the same dual critic head without
trust weighting.
"""

import torch
import torch.nn as nn
from stable_baselines3.ppo.policies import ActorCriticPolicy

LOG_VSTD_MIN = -5.0
LOG_VSTD_MAX = 3.0


class UnifiedPolicy(ActorCriticPolicy):
    """Policy supporting both MSE (single V head) and CW-VL (mu + log_var heads).

    The feature extractor backbone is identical for both modes.
    Only the final critic head(s) differ.
    """

    LOG_VSTD_MIN = LOG_VSTD_MIN
    LOG_VSTD_MAX = LOG_VSTD_MAX

    def __init__(
        self,
        *args,
        value_loss_type: str = "mse",
        freeze_sigma: bool = False,
        actor_learning_rate: float = 0.0,
        **kwargs,
    ):
        # value_loss_type must be set before super().__init__ because
        # SB3 calls _build() inside __init__ which calls _build_mlp_extractor()
        self._value_loss_type = value_loss_type
        self._freeze_sigma = freeze_sigma
        self._actor_learning_rate = float(actor_learning_rate)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Build MLP extractor (SB3 internal)."""
        super()._build_mlp_extractor()

    def _build(self, lr_schedule) -> None:
        """Build policy network, replacing the default value head with our configurable one(s).

        We call super()._build() first to create the default value_net + optimizer,
        then replace the value heads and recreate the optimizer to include the correct params.
        """
        import torch as th

        super()._build(lr_schedule)
        last_dim = self.mlp_extractor.latent_dim_vf

        if self._value_loss_type == "mse":
            # Replace with fresh single value head (re-init to avoid inheriting super's weights)
            self.value_net = nn.Linear(last_dim, 1)
            nn.init.orthogonal_(self.value_net.weight, gain=1.0)
            nn.init.constant_(self.value_net.bias, 0.0)
            self._has_logvar = False

        elif self._value_loss_type in ("hetero", "cwvl"):
            self.value_mu = nn.Linear(last_dim, 1)
            nn.init.orthogonal_(self.value_mu.weight, gain=1.0)
            nn.init.constant_(self.value_mu.bias, 0.0)

            self.value_logstd_base = nn.Linear(last_dim, 1)
            nn.init.zeros_(self.value_logstd_base.weight)
            nn.init.constant_(self.value_logstd_base.bias, 0.0)

            if self._freeze_sigma:
                with th.no_grad():
                    self.value_logstd_base.weight.zero_()
                    self.value_logstd_base.bias.zero_()
                for p in self.value_logstd_base.parameters():
                    p.requires_grad_(False)

            # value_net must exist for SB3 serialization compatibility
            self.value_net = self.value_mu
            self._has_logvar = True

        else:
            raise ValueError(f"Unknown value_loss_type: {self._value_loss_type}")

        # Recreate optimizer to include the new value head(s) and exclude the old one.
        # super()._build already set self.optimizer_class and self.optimizer_kwargs.
        params = self._optimizer_params()
        self.optimizer = self.optimizer_class(params, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _optimizer_params(self):
        base_params = []
        actor_params = []

        for name, param in self.named_parameters():
            if (
                self._actor_learning_rate > 0.0
                and (
                    name == "log_std"
                    or name.startswith("mlp_extractor.policy_net.")
                    or name.startswith("action_net.")
                )
            ):
                actor_params.append(param)
            else:
                base_params.append(param)

        if not actor_params:
            return self.parameters()

        groups = [{"params": base_params}]
        if actor_params:
            groups.append({"params": actor_params, "lr": self._actor_learning_rate, "actor_lr_group": True})
        return groups

    # ---- Critic forward ----

    def _compute_value(self, latent_vf: torch.Tensor):
        """Return (value, log_var_or_None)."""
        if not self._has_logvar:
            return self.value_net(latent_vf), None
        mu = self.value_mu(latent_vf)
        if self._freeze_sigma:
            log_std = torch.zeros_like(mu)
        else:
            log_std = self.value_logstd_base(latent_vf)
        log_std = log_std.clamp(LOG_VSTD_MIN, LOG_VSTD_MAX)
        return mu, log_std

    # ---- SB3 interface ----

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Called during rollout for GAE advantage computation.
        Returns scalar value predictions (mu for CW-VL)."""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        value, _ = self._compute_value(latent_vf)
        return value

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        value, _ = self._compute_value(latent_vf)
        return actions, value, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value, log_var = self._compute_value(latent_vf)
        return value, log_prob, entropy, log_var
