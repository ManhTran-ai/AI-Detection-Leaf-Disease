from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2


def _build_inference_transform(image_size: int, mean, std, pad_if_needed: bool = True):
    resize_steps = [A.LongestMaxSize(image_size)]
    if pad_if_needed:
        resize_steps.append(A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0))
    else:
        resize_steps = [A.Resize(image_size, image_size)]

    return A.Compose(
        [
            *resize_steps,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image {image_path} not found.")
    return np.array(Image.open(path).convert("RGB"))


def preprocess_image(image_path: str, dataset_cfg: Dict, augmentation_cfg: Optional[Dict] = None):
    image = load_image(image_path)
    aug_cfg = augmentation_cfg or {}
    transform = _build_inference_transform(
        dataset_cfg["image_size"],
        dataset_cfg["mean"],
        dataset_cfg["std"],
        pad_if_needed=aug_cfg.get("pad_if_needed", True),
    )
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0), image


