from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .model_factory import build_model


def _build_optimizer(model: nn.Module, cfg: Dict) -> optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.0)

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=True,
        )
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_model_components(config: Dict, device: torch.device) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))

    optimizer = _build_optimizer(model, config["training"])
    return model, optimizer, criterion


