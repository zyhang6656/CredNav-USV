"""Value loss functions for PPO baseline, Hetero-PPO, and CW-VL.

PPO baseline: MSE loss  L_V = 0.5 * (R_hat - V)^2
Hetero-PPO:    Gaussian NLL  L_V = 0.5 * (s + exp(-s) * (R - mu)^2)
CW-VL:         Tempered Gaussian NLL  L_V = 0.5 * t * (s + exp(-s) * (R - mu)^2)
"""

import torch as th


def mse_value_loss(
    values: th.Tensor,
    returns: th.Tensor,
    old_values: th.Tensor | None = None,
    clip_range: float | None = None,
) -> th.Tensor:
    """Standard PPO MSE value loss with optional clipping.

    Args:
        values: Predicted values V_phi(b), shape (N,).
        returns: Bootstrapped return targets R_hat, shape (N,).
        old_values: Previous value predictions (for clipping).
        clip_range: Value clipping range.

    Returns:
        Scalar loss.
    """
    values = values.flatten()
    returns = returns.flatten()

    if clip_range is None:
        return 0.5 * th.mean((returns - values) ** 2)

    old_values = old_values.flatten()
    v_clipped = old_values + th.clamp(values - old_values, -clip_range, clip_range)
    loss_unclipped = (returns - values) ** 2
    loss_clipped = (returns - v_clipped) ** 2
    return 0.5 * th.mean(th.max(loss_unclipped, loss_clipped))


def cwvl_value_loss(
    mu: th.Tensor,
    log_var: th.Tensor,
    returns: th.Tensor,
    trust: th.Tensor,
    old_mu: th.Tensor | None = None,
    clip_range: float | None = None,
    log_var_min: float = -5.0,
    log_var_max: float = 3.0,
) -> th.Tensor:
    """CW-VL tempered Gaussian NLL value loss.

    L_V = 0.5 * mean( t_i * (s_i + exp(-s_i) * (R_i - mu_i)^2) )

    Args:
        mu: Value mean predictions, shape (N,).
        log_var: Log-variance predictions s_phi(b), shape (N,).
        returns: Bootstrapped return targets R_hat, shape (N,).
        trust: Per-transition trust factors t(k), shape (N,).
        old_mu: Previous mu predictions (for clipping).
        clip_range: Value clipping range on mu.
        log_var_min: Clamp lower bound for log variance.
        log_var_max: Clamp upper bound for log variance.

    Returns:
        Scalar loss.
    """
    mu = mu.flatten()
    log_var = log_var.flatten()
    returns = returns.flatten()
    trust = trust.flatten()

    log_var = th.clamp(log_var, log_var_min, log_var_max)
    sigma_sq = th.exp(log_var).clamp(min=1e-6)

    def nll(v: th.Tensor) -> th.Tensor:
        return 0.5 * trust * (log_var + (returns - v) ** 2 / sigma_sq)

    if clip_range is None:
        return th.mean(nll(mu))

    old_mu = old_mu.flatten()
    mu_clipped = old_mu + th.clamp(mu - old_mu, -clip_range, clip_range)
    return th.mean(th.max(nll(mu), nll(mu_clipped)))


def hetero_value_loss(
    mu: th.Tensor,
    log_var: th.Tensor,
    returns: th.Tensor,
    old_mu: th.Tensor | None = None,
    clip_range: float | None = None,
    log_var_min: float = -5.0,
    log_var_max: float = 3.0,
) -> th.Tensor:
    """Unweighted Gaussian NLL using the same critic parameterization as CW-VL."""
    return cwvl_value_loss(
        mu,
        log_var,
        returns,
        th.ones_like(returns),
        old_mu=old_mu,
        clip_range=clip_range,
        log_var_min=log_var_min,
        log_var_max=log_var_max,
    )
