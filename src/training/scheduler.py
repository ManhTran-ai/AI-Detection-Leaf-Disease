from typing import Dict, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def create_scheduler(optimizer: Optimizer, scheduler_cfg: Dict):
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
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", 50),
            eta_min=scheduler_cfg.get("min_lr", 1e-6),
        )

    raise ValueError(f"Unsupported scheduler: {name}")


