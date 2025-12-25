from collections import Counter
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import DurianLeafDataset
from .preprocessing import build_eval_transforms, build_train_transforms


def _build_dataset(split: str, config: Dict):
    dataset_cfg = config["dataset"]
    transforms = build_train_transforms(config) if split == "train" else build_eval_transforms(config)
    split_dir = dataset_cfg[f"{split}_dir"]
    return DurianLeafDataset(
        root_dir=split_dir,
        class_names=dataset_cfg["class_names"],
        transforms=transforms,
        return_paths=split != "train",
    )


def _create_sampler(dataset: DurianLeafDataset):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_dataset = _build_dataset("train", config)
    val_dataset = _build_dataset("val", config)
    test_dataset = _build_dataset("test", config)

    loader_params = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": dataset_cfg.get("num_workers", 4),
        "pin_memory": dataset_cfg.get("pin_memory", True),
    }

    sampler = _create_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader, test_loader


