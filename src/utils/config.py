import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


@dataclass
class ExperimentConfig:
    """Light-weight wrapper so we can access nested config dictionaries safely."""

    raw: Dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self.raw
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ExperimentConfig(raw=data)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


