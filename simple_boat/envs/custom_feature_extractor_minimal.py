"""Feature extractor for minimal USV environment.

Consumes {state, dyn, dyn_mask} from the observation dict. The global t_min
trust value is not an observation key; trainers pass it through rollout info.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as DictSpace


class MinimalFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor that fuses state + masked dynamic-obstacle features.

    Global t_min is excluded from fused features. Per-target trust may still
    appear inside dyn when the environment provides it.
    """

    def __init__(self, observation_space: DictSpace, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        state_dim = observation_space.spaces["state"].shape[0]
        dyn_dim = observation_space.spaces["dyn"].shape[0]
        if dyn_dim % 7 != 0:
            raise ValueError(f"dyn_dim must be divisible by 7, got {dyn_dim}")
        self.n_dyn = dyn_dim // 7

        self.mlp_state = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mlp_dyn = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        combined_dim = 64 + 64  # state + dyn only; no global t_min, no colregs
        self.linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim), nn.ReLU(),
        )
        self._features_dim = features_dim

    def _dyn_mask(self, observations: dict, dyn: torch.Tensor) -> torch.Tensor:
        mask = observations.get("dyn_mask")
        if mask is None:
            return torch.ones(dyn.shape[0], self.n_dyn, 1, device=dyn.device, dtype=dyn.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return (mask.reshape(dyn.shape[0], self.n_dyn, 1) > 0.5).to(dtype=dyn.dtype, device=dyn.device)

    def _masked_dyn_features(self, observations: dict) -> torch.Tensor:
        dyn = observations["dyn"]
        dyn_rows = dyn.reshape(dyn.shape[0], self.n_dyn, 7)
        mask = self._dyn_mask(observations, dyn)
        slot_features = self.obstacle_encoder(dyn_rows) * mask
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = slot_features.sum(dim=1) / counts

        has_any = (mask.sum(dim=1) > 0.0)
        max_pool = slot_features.masked_fill(mask <= 0.0, -torch.inf).max(dim=1).values
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))
        return self.mlp_dyn(torch.cat([mean_pool, max_pool], dim=1))

    def forward(self, observations: dict) -> torch.Tensor:
        state_out = self.mlp_state(observations["state"])
        dyn_out = self._masked_dyn_features(observations)
        fused = torch.cat([state_out, dyn_out], dim=1)
        return self.linear(fused)
