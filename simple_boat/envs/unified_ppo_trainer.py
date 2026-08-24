"""Unified PPO trainer supporting PPO, Hetero-PPO, and CW-VL critic loss.

Usage:
  trainer = UnifiedPPOTrainer(..., value_loss_type="mse")   # PPO baseline
  trainer = UnifiedPPOTrainer(..., value_loss_type="hetero")  # Hetero-PPO
  trainer = UnifiedPPOTrainer(..., value_loss_type="cwvl")  # PPO + CW-VL

Key invariants (verified at init):
  - Actor observation {state, dyn, dyn_mask} is identical for both modes.
  - trust (t_min) captured from env info, injected into buffer AFTER rollout.
  - Global rollout trust is used only by the CW-VL critic loss.
  - Same rollout, reward, and PPO update hyperparams across modes.
"""

import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from simple_boat.algorithms.ppo.value_losses import hetero_value_loss


def cwvl_critic_trust_weights(
    trust: th.Tensor,
    power: float = 1.0,
    normalize: bool = False,
) -> th.Tensor:
    trust = th.clamp(trust.float().flatten(), 1e-4, 1.0)
    weights = trust.pow(max(float(power), 0.0))
    if normalize:
        weights = weights / weights.mean().clamp(min=1e-6).detach()
    return weights


def apply_optimizer_group_lr(optimizer: th.optim.Optimizer, marker: str, lr: float) -> bool:
    if float(lr) <= 0.0:
        return False
    changed = False
    for group in optimizer.param_groups:
        if group.get(marker, False):
            group["lr"] = float(lr)
            changed = True
    return changed


class UnifiedPPOTrainer(PPO):
    """PPO trainer with switchable MSE or Gaussian NLL value loss.

    PPO baseline (value_loss_type="mse"):
      L_V = 0.5 * mean((R_hat - V)^2)

    CW-VL (value_loss_type="cwvl"):
      L_V = 0.5 * t_i * (s_i + exp(-s_i) * (R_i - mu_i)^2)   per-transition

    Hetero-PPO (value_loss_type="hetero") uses the same NLL with t_i = 1.
    """

    def __init__(
        self,
        *args,
        value_loss_type: str = "mse",
        cwvl_trust_power: float = 1.0,
        cwvl_normalize_trust_weights: bool = False,
        actor_learning_rate: float = 0.0,
        **kwargs,
    ):
        if value_loss_type not in ("mse", "hetero", "cwvl"):
            raise ValueError(f"value_loss_type must be 'mse', 'hetero', or 'cwvl', got '{value_loss_type}'")
        self._value_loss_type = value_loss_type
        self._cwvl_trust_power = float(cwvl_trust_power)
        self._cwvl_normalize_trust_weights = bool(cwvl_normalize_trust_weights)
        self._actor_learning_rate = float(actor_learning_rate)
        self._verification_logged = False
        # Accumulator for trust values captured during rollout
        self._collected_trusts: list[float] = []
        super().__init__(*args, **kwargs)

    def _log_verification(self) -> None:
        if self._verification_logged:
            return
        self._verification_logged = True

        obs_space = self.env.observation_space
        if hasattr(obs_space, "spaces"):
            obs_keys = list(obs_space.spaces.keys())
        else:
            obs_keys = ["<flat>"]

        # Count parameters
        policy = self.policy
        actor_params = 0
        critic_trunk_params = 0
        logvar_params = 0
        for name, param in policy.named_parameters():
            n = param.numel()
            if "mlp_extractor" in name or "features_extractor" in name:
                # shared trunk
                pass
            elif "action_net" in name or "log_std" in name:
                actor_params += n
            elif "value_logstd" in name:
                logvar_params += n
            elif "value_net" in name or "value_mu" in name:
                critic_trunk_params += n

        print("=" * 60)
        print("[VERIFY] UnifiedPPOTrainer:")
        print(f"  value_loss_type          = {self._value_loss_type}")
        print(f"  env obs keys             = {obs_keys}")
        print(f"  global trust in actor    = False")
        print(f"  global trust in critic   = False")
        print(f"  trust used in value loss = {self._value_loss_type == 'cwvl'}")
        print(f"  actor loss weighted      = False")
        print(f"  trust power in CW-VL     = {getattr(self, '_cwvl_trust_power', 1.0) if self._value_loss_type == 'cwvl' else 0.0}")
        print(f"  normalize trust weights  = {getattr(self, '_cwvl_normalize_trust_weights', False) if self._value_loss_type == 'cwvl' else False}")
        print(f"  actor group lr           = {getattr(self, '_actor_learning_rate', 0.0)}")
        print(f"  action post-process      = False (shield=OFF)")
        print(f"  n_steps={self.n_steps} batch={self.batch_size} epochs={self.n_epochs}")
        print(f"  gamma={self.gamma} lambda={self.gae_lambda} clip={self.clip_range}")
        print(f"  ent={self.ent_coef} vf={self.vf_coef} lr={self.learning_rate}")
        print(f"  CW-VL extra params (log_var head) = {logvar_params}")
        print("=" * 60)

    # ==================================================================
    #   collect_rollouts override — capture trust from info
    # ==================================================================

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3-compatible rollout collection with trust capture from env info.

        After collection, trust values are injected into the buffer's
        observations dict as key '_trust'. This key:
          - Is NOT present in env observation_space.
          - Is NOT seen by the feature extractor as a private key.
          - Is read ONLY by train() for CW-VL critic loss.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        self._collected_trusts.clear()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            # --- Capture trust from env info ---
            for info in infos:
                t_val = float(info.get("t_min", 1.0))
                self._collected_trusts.append(t_val)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            # Ensure actions are numpy array (already batched from policy)
            actions = np.array(actions) if not isinstance(actions, np.ndarray) else actions

            # Handle episode end
            for idx, done in enumerate(dones):
                if done:
                    if callback.on_rollout_end() is False:
                        return False

            rollout_buffer.add(
                self._last_obs, actions, rewards,
                self._last_episode_starts, values, log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # --- Inject collected trusts into buffer observations ---
        self._inject_trusts_into_buffer(rollout_buffer)

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(values, dones)

        callback.on_rollout_end()
        return True

    def _inject_trusts_into_buffer(self, rollout_buffer) -> None:
        """Add '_trust' key to buffer observations after rollout collection.

        SB3 buffer layout: (n_steps, n_envs, *obs_shape).
        Collected trusts are per-env-step, stored sequentially during rollout.
        We reshape into (n_steps, n_envs, 1) to match buffer layout.
        """
        n_envs = rollout_buffer.n_envs
        n_steps = rollout_buffer.buffer_size
        n_collected = len(self._collected_trusts)

        if n_collected < n_steps * n_envs:
            # Pad with 1.0 for any missing steps
            trusts_arr = np.ones(n_steps * n_envs, dtype=np.float32)
            trusts_arr[:n_collected] = np.array(self._collected_trusts, dtype=np.float32)
        else:
            trusts_arr = np.array(self._collected_trusts[:n_steps * n_envs], dtype=np.float32)

        rollout_buffer.observations["_trust"] = trusts_arr.reshape(n_steps, n_envs, 1)

    # ==================================================================
    #   train() — CW-VL reads trust from buffer observations["_trust"]
    # ==================================================================

    def train(self) -> None:
        self._log_verification()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        apply_optimizer_group_lr(
            self.policy.optimizer,
            "actor_lr_group",
            getattr(self, "_actor_learning_rate", 0.0),
        )
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        approx_kls, clip_fractions = [], []

        value_mse_list, value_abs_mean_list = [], []
        logvar_mean_list, logvar_std_list = [], []
        logvar_clamp_frac_list = []
        trust_mean_list = []
        trust_weight_mean_list = []

        continue_training = True
        n_optimizer_steps = 0

        for epoch in range(self.n_epochs):
            if not continue_training:
                break

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # Extract trust before stripping (CW-VL only)
                if self._value_loss_type == "cwvl":
                    t_min_vals = rollout_data.observations.get("_trust",
                        th.ones(actions.shape[0], 1, device=actions.device))
                else:
                    t_min_vals = None

                # Strip _trust key — not in observation_space, would crash preprocess_obs
                obs_for_policy = {k: v for k, v in rollout_data.observations.items()
                                  if not k.startswith("_")}

                if self._value_loss_type != "mse":
                    values, log_prob, entropy, value_log_var = self.policy.evaluate_actions(
                        obs_for_policy, actions,
                    )
                    values = values.flatten()
                    value_log_var = value_log_var.flatten()
                else:
                    values, log_prob, entropy, _ = self.policy.evaluate_actions(
                        obs_for_policy, actions,
                    )
                    values = values.flatten()

                log_prob = log_prob.flatten()
                old_log_prob = rollout_data.old_log_prob.flatten()
                returns = rollout_data.returns.flatten()

                # Policy loss.
                ratio = th.exp(log_prob - old_log_prob)
                advantages = rollout_data.advantages.flatten().detach()
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_surrogate = th.min(policy_loss_1, policy_loss_2)
                policy_loss = -policy_surrogate.mean()

                clip_fraction = th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if entropy is None:
                    entropy_loss = th.mean(log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # Critic value loss
                if self._value_loss_type == "mse":
                    if clip_range_vf is None:
                        value_loss = 0.5 * th.mean((returns - values) ** 2)
                    else:
                        old_values = rollout_data.old_values.flatten()
                        v_clipped = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                        value_loss = 0.5 * th.mean(th.max(
                            (returns - values) ** 2, (returns - v_clipped) ** 2,
                        ))
                else:
                    logvar_min = getattr(self.policy, "LOG_VSTD_MIN", -5.0)
                    logvar_max = getattr(self.policy, "LOG_VSTD_MAX", 3.0)
                    value_log_var = th.clamp(value_log_var, logvar_min, logvar_max)
                    if self._value_loss_type == "cwvl":
                        t_min = th.clamp(t_min_vals.float().flatten(), 1e-4, 1.0)
                        trust_weights = cwvl_critic_trust_weights(
                            t_min,
                            power=getattr(self, "_cwvl_trust_power", 1.0),
                            normalize=getattr(self, "_cwvl_normalize_trust_weights", False),
                        )
                        sigma_sq = th.exp(value_log_var).clamp(min=1e-6)

                        def nll(v: th.Tensor) -> th.Tensor:
                            return 0.5 * trust_weights * (value_log_var + (returns - v) ** 2 / sigma_sq)

                        if clip_range_vf is None:
                            value_loss = th.mean(nll(values))
                        else:
                            old_values = rollout_data.old_values.flatten()
                            v_clipped = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                            value_loss = th.mean(th.max(nll(values), nll(v_clipped)))
                    else:
                        value_loss = hetero_value_loss(
                            values,
                            value_log_var,
                            returns,
                            old_mu=(rollout_data.old_values.flatten() if clip_range_vf is not None else None),
                            clip_range=clip_range_vf,
                            log_var_min=logvar_min,
                            log_var_max=logvar_max,
                        )

                    with th.no_grad():
                        value_mse_list.append(th.mean((returns - values) ** 2).item())
                        value_abs_mean_list.append(th.abs(returns - values).mean().item())
                        logvar_mean_list.append(value_log_var.mean().item())
                        logvar_std_list.append(value_log_var.std(unbiased=False).item())
                        hit = ((value_log_var <= logvar_min + 1e-6) | (value_log_var >= logvar_max - 1e-6)).float().mean().item()
                        logvar_clamp_frac_list.append(hit)
                        if self._value_loss_type == "cwvl":
                            trust_mean_list.append(t_min.mean().item())
                            trust_weight_mean_list.append(trust_weights.mean().item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                approx_kls.append(approx_kl)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                n_optimizer_steps += 1

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        self._n_updates += n_optimizer_steps

        def _avg(x):
            return float(sum(x) / max(len(x), 1))

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("train/entropy_loss", _avg(entropy_losses))
            self.logger.record("train/policy_gradient_loss", _avg(pg_losses))
            self.logger.record("train/value_loss", _avg(value_losses))
            self.logger.record("train/approx_kl", _avg(approx_kls))
            self.logger.record("train/clip_fraction", _avg(clip_fractions))

            try:
                ev = explained_variance(
                    self.rollout_buffer.values.flatten(),
                    self.rollout_buffer.returns.flatten(),
                )
            except Exception:
                ev = float("nan")
            self.logger.record("train/explained_variance", float(ev))

            if self._value_loss_type != "mse":
                self.logger.record("train/value_MSE", _avg(value_mse_list))
                self.logger.record("train/value_abs_error_mean", _avg(value_abs_mean_list))
                self.logger.record("train/value_log_var_mean", _avg(logvar_mean_list))
                self.logger.record("train/value_log_var_std", _avg(logvar_std_list))
                self.logger.record("train/value_log_var_clamp_frac", _avg(logvar_clamp_frac_list))
                if self._value_loss_type == "cwvl":
                    self.logger.record("train/trust_mean", _avg(trust_mean_list))
                    self.logger.record("train/trust_weight_mean", _avg(trust_weight_mean_list))
