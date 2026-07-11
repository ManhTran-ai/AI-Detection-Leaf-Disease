from typing import Dict, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Implements: warmup_epochs linear warmup → cosine annealing to min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine = 0.5 * (1 + cosine_decay(progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine
            for base_lr in self.base_lrs
        ]


def cosine_decay(x: float) -> float:
    import math
    return math.cos(math.pi * min(x, 1.0))


def create_scheduler(optimizer: Optimizer, scheduler_cfg: Dict, num_epochs: Optional[int] = None):
    name = scheduler_cfg.get("name", "none").lower()
    if name in {"none", "", None}:
        return None

    if name == "step":
        return StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )

    if name == "cosine":
        warmup = scheduler_cfg.get("warmup_epochs", 0)
        min_lr = scheduler_cfg.get("min_lr", 1e-6)
        total = num_epochs or scheduler_cfg.get("t_max", 50)
        if warmup > 0:
            return WarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup,
                total_epochs=total,
                min_lr=min_lr,
            )
        return CosineAnnealingLR(
            optimizer,
            T_max=total,
            eta_min=min_lr,
        )

    raise ValueError(f"Unsupported scheduler: {name}")


