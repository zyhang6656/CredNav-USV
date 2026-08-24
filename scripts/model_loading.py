"""Model loading helpers for UnifiedPPOTrainer checkpoints."""

from __future__ import annotations

from typing import Any, Optional


def _checkpoint_policy_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    policy_kwargs["value_loss_type"] = str(data.get("_value_loss_type", "mse"))
    if "freeze_sigma" in data:
        policy_kwargs["freeze_sigma"] = bool(data["freeze_sigma"])
    if float(data.get("_actor_learning_rate", 0.0) or 0.0) > 0.0:
        policy_kwargs["actor_learning_rate"] = float(data["_actor_learning_rate"])
    return policy_kwargs


def load_unified_policy(path: str, *, env: Optional[Any] = None, device: str = "auto"):
    """Load PPO/CW-VL checkpoints with the saved value-head structure."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from simple_boat.envs.unified_ppo_trainer import UnifiedPPOTrainer

    data, _, _ = load_from_zip_file(path, device=device, print_system_info=False)
    policy_kwargs = _checkpoint_policy_kwargs(data)

    model = UnifiedPPOTrainer.load(
        path,
        env=env,
        device=device,
        custom_objects={"policy_kwargs": policy_kwargs},
    )
    return model
