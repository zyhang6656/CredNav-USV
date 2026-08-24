import torch

from simple_boat.envs.unified_ppo_trainer import (
    apply_optimizer_group_lr,
    cwvl_critic_trust_weights,
)


def test_cwvl_critic_trust_weights_default_matches_plain_trust():
    trust = torch.tensor([0.2, 0.5, 1.0])

    weights = cwvl_critic_trust_weights(trust, power=1.0, normalize=False)

    assert torch.allclose(weights, trust)


def test_cwvl_critic_trust_weights_power_sharpens_low_trust_discount():
    trust = torch.tensor([0.2, 0.5, 1.0])

    weights = cwvl_critic_trust_weights(trust, power=2.0, normalize=False)

    assert torch.allclose(weights, torch.tensor([0.04, 0.25, 1.0]))


def test_cwvl_critic_trust_weights_normalization_preserves_batch_scale():
    trust = torch.tensor([0.25, 1.0])

    weights = cwvl_critic_trust_weights(trust, power=2.0, normalize=True)

    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[1] > weights[0]


def test_apply_optimizer_group_lr_restores_actor_group_lr():
    base_param = torch.nn.Parameter(torch.ones(1))
    actor_param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([
        {"params": [base_param], "lr": 2e-6},
        {"params": [actor_param], "lr": 2e-6, "actor_lr_group": True},
    ])

    changed = apply_optimizer_group_lr(optimizer, "actor_lr_group", 1e-4)

    assert changed is True
    assert optimizer.param_groups[0]["lr"] == 2e-6
    assert optimizer.param_groups[1]["lr"] == 1e-4
