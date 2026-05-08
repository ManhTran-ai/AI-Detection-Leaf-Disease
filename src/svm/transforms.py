"""Transforms for CNN feature extraction pipeline.

Provides simple, standalone transform composition that does NOT depend on
albumentations autoaugment (which adds heavy augmentation for training).
Feature extraction only needs resize + normalize.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from .config import IMAGE_MEAN, IMAGE_STD


def get_feature_extraction_transform(image_size: int) -> A.Compose:
    """Build a simple evaluation transform for feature extraction.

    Resizes to the model's expected image size, pads if needed, normalizes
    with ImageNet mean/std, and converts to tensor.

    Args:
        image_size: Target square image size (e.g. 224 for most models, 256 for ResNet50).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])
