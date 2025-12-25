from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2

from .autoaugment import get_auto_augment


def _resize_and_pad(dataset_cfg: Dict, aug_cfg: Dict) -> List:
    """Aspect-ratio-preserving resize with optional padding to a square canvas."""
    image_size = dataset_cfg["image_size"]
    if aug_cfg.get("pad_if_needed", True):
        return [
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]
    return [A.Resize(image_size, image_size)]


def _base_transforms(dataset_cfg: Dict, aug_cfg: Dict) -> List:
    """Base deterministic transforms shared by train/eval/inference."""
    return [
        *_resize_and_pad(dataset_cfg, aug_cfg),
        A.Normalize(mean=dataset_cfg["mean"], std=dataset_cfg["std"]),
        ToTensorV2(),
    ]


class AutoAugmentAlbumentations:
    """Albumentations-compatible wrapper for AutoAugment transforms.

    This class wraps PIL-based AutoAugment policies to work with
    the albumentations pipeline (which uses numpy arrays).

    Args:
        policy: AutoAugment policy name ("imagenet", "rand", "trivial")
        n: Number of operations for RandAugment
        m: Magnitude for RandAugment (0-10)
        p: Probability of applying augmentation
    """

    def __init__(self, policy: str = "rand", n: int = 2, m: int = 9, p: float = 1.0):
        self.policy = policy
        self.n = n
        self.m = m
        self.p = p
        self.transform = get_auto_augment(policy, n, m)

    def __call__(self, force_apply: bool = False, **data) -> dict:
        """Apply augmentation to image data."""
        if "image" not in data:
            return data

        image = data["image"]

        if force_apply or np.random.random() < self.p:
            # Convert numpy to PIL
            pil_image = Image.fromarray(image)
            # Apply auto augment
            pil_image = self.transform(pil_image)
            # Convert back to numpy
            image = np.array(pil_image)
            data["image"] = image

        return data

    def __repr__(self):
        return f"AutoAugmentAlbumentations(policy={self.policy}, n={self.n}, m={self.m}, p={self.p})"


def build_train_transforms(config: Dict) -> A.Compose:
    """Build training transforms including optional AutoAugmentation.

    Supports three AutoAugment policies:
    - "imagenet": Original AutoAugment policy from the paper
    - "rand": RandAugment (simpler, often equally effective)
    - "trivial": TrivialAugment (one random op, no tuning needed)

    Config augmentation options:
    - auto_augment: "imagenet" | "rand" | "trivial" | null
    - rand_augment_n: Number of operations (default: 2)
    - rand_augment_m: Magnitude 0-10 (default: 9)
    """
    aug_cfg = config.get("augmentation", {})
    dataset_cfg = config["dataset"]
    transforms = []

    # AutoAugment (applied first, before other augmentations)
    auto_augment_policy = aug_cfg.get("auto_augment", None)
    if auto_augment_policy:
        rand_n = aug_cfg.get("rand_augment_n", 2)
        rand_m = aug_cfg.get("rand_augment_m", 9)
        auto_aug = AutoAugmentAlbumentations(
            policy=auto_augment_policy,
            n=rand_n,
            m=rand_m,
            p=aug_cfg.get("auto_augment_p", 1.0),
        )
        transforms.append(auto_aug)

    # Optional conservative random crop (disabled by default)
    if aug_cfg.get("use_tight_crop", False):
        transforms.append(
            A.RandomResizedCrop(
                height=dataset_cfg["image_size"],
                width=dataset_cfg["image_size"],
                scale=tuple(aug_cfg.get("tight_crop_scale", [0.9, 1.0])),
                ratio=tuple(aug_cfg.get("tight_crop_ratio", [0.9, 1.1])),
            )
        )
    else:
        transforms.extend(_resize_and_pad(dataset_cfg, aug_cfg))

        if aug_cfg.get("use_center_crop", False):
            crop_pct = aug_cfg.get("center_crop_pct", 0.9)
            crop_size = max(1, min(dataset_cfg["image_size"], int(dataset_cfg["image_size"] * crop_pct)))
            transforms.append(A.CenterCrop(crop_size, crop_size))
            transforms.append(A.Resize(dataset_cfg["image_size"], dataset_cfg["image_size"]))
        else:
            crop_margin_pct = max(0.0, aug_cfg.get("crop_margin_pct", 0.0))
            if crop_margin_pct > 0:
                crop_size = int(dataset_cfg["image_size"] * (1 - 2 * crop_margin_pct))
                if crop_size > 0:
                    transforms.append(A.RandomCrop(crop_size, crop_size))
                    transforms.append(A.Resize(dataset_cfg["image_size"], dataset_cfg["image_size"]))

    rotation = aug_cfg.get("rotation", 0)
    if rotation:
        transforms.append(A.Rotate(limit=rotation, p=0.8))

    if aug_cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.2))

    if any(aug_cfg.get(k, 0) > 0 for k in ["brightness", "contrast", "saturation", "hue"]):
        transforms.append(
            A.ColorJitter(
                brightness=aug_cfg.get("brightness", 0),
                contrast=aug_cfg.get("contrast", 0),
                saturation=aug_cfg.get("saturation", 0),
                hue=aug_cfg.get("hue", 0),
                p=0.8,
            )
        )

    if aug_cfg.get("blur_prob", 0) > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg["blur_prob"]))

    transforms.extend(
        [
            A.Normalize(mean=dataset_cfg["mean"], std=dataset_cfg["std"]),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def build_eval_transforms(config: Dict) -> A.Compose:
    dataset_cfg = config["dataset"]
    aug_cfg = config.get("augmentation", {})
    return A.Compose(_base_transforms(dataset_cfg, aug_cfg))


