from dataclasses import dataclass
from typing import Optional

import torch

from src.models.utils import save_checkpoint


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"


class EarlyStopping:
    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> None:
        if self.best_score is None or metric < self.best_score - self.config.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.config.patience:
                self.should_stop = True


class CheckpointManager:
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = float("inf")

    def save_best(self, model, optimizer, epoch: int, metric: float, history) -> Optional[str]:
        if metric < self.best_metric:
            self.best_metric = metric
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": metric,
                "history": history,
            }
            return save_checkpoint(state, self.checkpoint_dir)
        return None


