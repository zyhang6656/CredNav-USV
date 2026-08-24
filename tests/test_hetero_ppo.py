from pathlib import Path
import numpy as np
import torch
import yaml
from gymnasium import spaces

from scripts.train_obs3 import resolve_value_loss_type
from simple_boat.algorithms.ppo.value_losses import cwvl_value_loss, hetero_value_loss
from simple_boat.envs.custom_ppo_policy_unified import UnifiedPolicy


def test_hetero_ppo_config_resolves_to_hetero_value_loss():
    assert resolve_value_loss_type({"algo": "hetero_ppo"}) == "hetero"


def test_hetero_loss_matches_cwvl_with_unit_trust():
    mu = torch.tensor([0.5, -0.5])
    log_var = torch.tensor([0.2, -0.3])
    returns = torch.tensor([1.0, -1.5])

    expected = cwvl_value_loss(mu, log_var, returns, torch.ones_like(returns))

    assert torch.allclose(hetero_value_loss(mu, log_var, returns), expected)


def test_hetero_policy_uses_dual_head():
    obs_space = spaces.Dict({
        "state": spaces.Box(-10.0, 10.0, shape=(8,), dtype=np.float32),
        "dyn": spaces.Box(-10.0, 10.0, shape=(42,), dtype=np.float32),
    })
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    policy = UnifiedPolicy(
        obs_space,
        action_space,
        lambda _: 1e-3,
        value_loss_type="hetero",
    )

    assert policy._has_logvar is True
    assert hasattr(policy, "value_mu")
    assert hasattr(policy, "value_logstd_base")


def test_hetero_5m_config_matches_cwvl_training_parameters():
    root = Path(__file__).resolve().parents[1]
    ppo = yaml.safe_load((root / "configs/experiments/mixed/mixed_obs3_6_delay20_cov100_binned_s10_5m_ppo_masked.yaml").read_text(encoding="utf-8"))
    cwvl = yaml.safe_load((root / "configs/experiments/mixed/mixed_obs3_6_delay20_cov100_binned_s10_5m_cwvl_critic_only.yaml").read_text(encoding="utf-8"))
    hetero = yaml.safe_load((root / "configs/experiments/mixed/mixed_obs3_6_delay20_cov100_binned_s10_5m_hetero_ppo.yaml").read_text(encoding="utf-8"))

    for config in (ppo, cwvl, hetero):
        config.pop("experiment_name")
        config.pop("algo")
        config.pop("run")

    assert hetero == cwvl == ppo
