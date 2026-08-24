"""Training-metrics callback used by the public trainer."""

import csv
import numpy as np
import pathlib
from collections import defaultdict
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class TrainingMetricsCallback(BaseCallback):
    """Logs training metrics (losses, entropy, etc.) to CSV for later plotting."""

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = pathlib.Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def _on_step(self) -> bool:
        # SB3 logs training metrics to logger every rollout
        # We collect them at each call
        if self.logger is not None and hasattr(self.logger, 'name_to_value'):
            values = self.logger.name_to_value
            train_metrics = {k: v for k, v in values.items()
                           if any(p in k for p in ['train/', 'loss', 'entropy', 'kl', 'clip',
                                                    'explained_variance', 'learning_rate'])}
            if train_metrics:
                row = {'timestep': self.num_timesteps}
                row.update({k.replace('train/', ''): v for k, v in train_metrics.items()})

                file_exists = self.csv_path.exists()
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not file_exists or not self._header_written:
                        writer.writeheader()
                        self._header_written = True
                    writer.writerow(row)
        return True
