"""Detection dataloaders for YOLO format."""

from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import YOLODataset


def build_detection_train_transform(img_size: int = 640) -> A.Compose:
    """Build augmentation pipeline for detection training.

    Args:
        img_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.Blur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def build_detection_val_transform(img_size: int = 640) -> A.Compose:
    """Build preprocessing pipeline for detection validation.

    Args:
        img_size: Target image size.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
            A.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


class DetectionDataLoader:
    """DataLoader wrapper that handles YOLO-format detection data with albumentations."""

    def __init__(
        self,
        dataset: YOLODataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        collate_fn=None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn or self._collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self) -> int:
        return len(self._dataloader)

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for variable-size detection batches."""
        max_h = max(item["image"].shape[0] for item in batch)
        max_w = max(item["image"].shape[1] for item in batch)
        batch_size = len(batch)

        images = []
        boxes_list = []
        labels_list = []
        paths = []
        orig_sizes = []

        for item in batch:
            img = item["image"]
            h, w = img.shape[:2]
            if h < max_h or w < max_w:
                img = np.pad(img, ((0, max_h - h), (0, max_w - w), (0, 0)), mode="constant", constant_values=114)

            images.append(img)
            boxes_list.append(item["boxes"])
            labels_list.append(item["labels"])
            paths.append(item["image_path"])
            orig_sizes.append(item["orig_size"])

        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

        return {
            "images": images,
            "boxes": boxes_list,
            "labels": labels_list,
            "paths": paths,
            "orig_sizes": orig_sizes,
        }


def create_detection_loaders(
    root_dir: str,
    class_names: Sequence[str],
    batch_size: int = 16,
    img_size: int = 640,
    num_workers: int = 4,
    train_dir: str = "train",
    val_dir: str = "val",
    test_dir: str = "test",
) -> Tuple[DetectionDataLoader, DetectionDataLoader, DetectionDataLoader]:
    """Create train/val/test dataloaders for object detection.

    Args:
        root_dir: Root directory of YOLO dataset.
        class_names: List of class names.
        batch_size: Batch size per loader.
        img_size: Target image size.
        num_workers: Number of data loading workers.
        train_dir: Name of train split folder.
        val_dir: Name of val split folder.
        test_dir: Name of test split folder.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = YOLODataset(
        root_dir=root_dir,
        split=train_dir,
        class_names=class_names,
        img_size=img_size,
        transform=build_detection_train_transform(img_size),
        augment=True,
    )
    val_dataset = YOLODataset(
        root_dir=root_dir,
        split=val_dir,
        class_names=class_names,
        img_size=img_size,
        transform=build_detection_val_transform(img_size),
        augment=False,
    )
    test_dataset = YOLODataset(
        root_dir=root_dir,
        split=test_dir,
        class_names=class_names,
        img_size=img_size,
        transform=build_detection_val_transform(img_size),
        augment=False,
    )

    train_loader = DetectionDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DetectionDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DetectionDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
