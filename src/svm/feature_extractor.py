"""CNN Feature Extractor — extract deep features from fine-tuned CNN models.

Loads a trained CNN checkpoint, removes the classification head, and extracts
feature vectors from the GlobalAveragePooling layer. Features are cached to
`.npy` files to avoid repeated extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    FEATURE_CACHE_DIR,
    IMAGE_MEAN,
    IMAGE_STD,
    IMG_EXTENSIONS,
    ModelInfo,
    get_model_info,
)
from .transforms import get_feature_extraction_transform

logger = logging.getLogger(__name__)


class _ImageDataset(Dataset):
    """Simple dataset that loads images from a list of paths and returns tensors."""

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        image_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [0] * len(image_paths)
        self.image_size = image_size
        self.transform = get_feature_extraction_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}. Returning zeros.")
            image = Image.new("RGB", (self.image_size, self.image_size))

        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        tensor = transformed["image"]
        return tensor, label


class _ImagePathDataset(Dataset):
    """Dataset that returns image paths and labels (no transform applied here)."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        image_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = get_feature_extraction_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[index]
        label = self.labels[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed["image"], label, image_path


def _collect_image_paths_and_labels(
    split_dir: str,
    class_names: List[str],
) -> Tuple[List[str], List[int]]:
    """Walk through split directory and collect all image paths with labels."""
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    paths: List[str] = []
    labels: List[int] = []

    split_path = Path(split_dir)
    for cls_name in class_names:
        cls_dir = split_path / cls_name
        if not cls_dir.exists():
            logger.warning(f"Class directory not found: {cls_dir}")
            continue
        for img_path in cls_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTENSIONS:
                paths.append(str(img_path))
                labels.append(class_to_idx[cls_name])

    if not paths:
        raise RuntimeError(f"No images found in {split_dir}")

    logger.info(f"Collected {len(paths)} images from {split_dir}")
    return paths, labels


def _replace_classifier_head(model: nn.Module, model_info: ModelInfo) -> nn.Module:
    """Replace the classification head with nn.Identity() for feature extraction."""
    classifier_attr = model_info.classifier_attr

    if classifier_attr == "fc":
        model.fc = nn.Identity()
    elif classifier_attr == "classifier":
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown classifier attribute: {classifier_attr}")

    return model


class CNNFeatureExtractor:
    """Extract deep features from fine-tuned CNN models.

    Loads a CNN from a `.pth` checkpoint, removes the classification head,
    and extracts feature vectors through batch inference. Results are cached
    to `.npy` files.

    Usage:
        extractor = CNNFeatureExtractor(model_name="resnet50")
        X_train = extractor.extract_from_dataset(train_dir, cache_tag="train")
        X_test = extractor.extract_from_dataset(test_dir, cache_tag="test")

    Attributes:
        model_name: Short model identifier (e.g. "resnet50").
        model_info: Full ModelInfo dataclass from config.
        device: torch.device used for inference.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        pretrained: bool = False,
    ):
        """
        Args:
            model_name: Model key from MODEL_REGISTRY (e.g. "resnet50").
            device: torch.device. Defaults to CUDA if available.
            pretrained: If True, load pretrained ImageNet weights instead of
                        fine-tuned checkpoint. Useful for ablation studies.
        """
        self.model_name = model_name
        self.model_info = get_model_info(model_name)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.pretrained = pretrained

        self._model: Optional[nn.Module] = None
        self._feature_dim: Optional[int] = None

        logger.info(
            f"CNNFeatureExtractor initialized: model={model_name}, "
            f"pretrained={pretrained}, device={self.device}"
        )

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._feature_dim = self.model_info.feature_dim
        return self._feature_dim

    def _build_model(self) -> nn.Module:
        """Build the CNN model, load checkpoint, and replace classifier head."""
        from src.models.model_factory import build_model
        from src.utils.config import load_config

        model_cfg = load_config(self.model_info.config_path).raw
        model = build_model(model_cfg).to(self.device)

        if not self.pretrained and self.model_info.checkpoint_exists():
            checkpoint = torch.load(
                self.model_info.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint: {self.model_info.checkpoint_path}")
        elif not self.pretrained:
            logger.warning(
                f"Checkpoint not found: {self.model_info.checkpoint_path}. "
                "Using pretrained weights."
            )
        else:
            logger.info("Using pretrained ImageNet weights (pretrained=True)")

        model = _replace_classifier_head(model, self.model_info)
        model.eval()
        return model

    @property
    def model(self) -> nn.Module:
        """Lazy-load the model on first access."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def _extract_batch(self, dataloader: DataLoader) -> np.ndarray:
        """Run batch inference to extract features."""
        all_features: List[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting features ({self.model_name})"):
                images = batch[0].to(self.device)
                features = self.model(images)
                all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

    def _extract_from_paths(
        self,
        image_paths: List[str],
        labels: List[int],
        cache_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Internal: extract features from a list of image paths."""
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached features from {cache_path}")
            features = np.load(cache_path)
            logger.info(f"Loaded features shape: {features.shape}")
            return features, labels

        dataset = _ImagePathDataset(
            image_paths=image_paths,
            labels=labels,
            image_size=self.model_info.image_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        features = self._extract_batch(dataloader)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            logger.info(f"Cached features to {cache_path} — shape: {features.shape}")

        return features, labels

    def extract_from_dataset(
        self,
        split_dir: str,
        class_names: Optional[List[str]] = None,
        cache_tag: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all images in a dataset split directory.

        Args:
            split_dir: Path to train/val/test directory containing class subfolders.
            class_names: List of class names matching subfolder names.
                        Defaults to the project's standard 5 disease classes.
            cache_tag: Optional tag to name the cache file, e.g. "train", "test".
                      Cache file: features/{model_name}_{cache_tag}_features.npy
                      If cache exists, load it instead of re-extracting.

        Returns:
            (features, labels) — both as numpy arrays.
        """
        from .config import DISEASE_CLASSES

        if class_names is None:
            class_names = DISEASE_CLASSES

        paths, labels = _collect_image_paths_and_labels(split_dir, class_names)

        cache_path = None
        if cache_tag:
            cache_path = str(FEATURE_CACHE_DIR / f"{self.model_name}_{cache_tag}_features.npy")

        features, labels = self._extract_from_paths(paths, labels, cache_path)
        return features, np.array(labels)

    def extract(
        self,
        image_paths: List[str],
        labels: List[int],
        cache_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from an explicit list of image paths.

        Args:
            image_paths: List of image file paths.
            labels: List of integer labels (0–4) for each image.
            cache_path: Optional path to cache file. If exists, load from cache.

        Returns:
            (features, labels) — both as numpy arrays.
        """
        features, labels = self._extract_from_paths(image_paths, labels, cache_path)
        return features, np.array(labels)

    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract features from a single image.

        Args:
            image_path: Path to a single image file.

        Returns:
            Feature vector as 1D numpy array of shape (feature_dim,).
        """
        dataset = _ImageDataset(
            image_paths=[image_path],
            labels=[0],
            image_size=self.model_info.image_size,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                features = self.model(images)
                return features.cpu().numpy()[0]
        raise RuntimeError("No features extracted")


def extract_all_splits(
    model_name: str,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to extract features from all three dataset splits.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) numpy arrays.
    """
    from .config import get_dataset_paths

    extractor = CNNFeatureExtractor(model_name, device=device)
    paths = get_dataset_paths()

    X_train, y_train = extractor.extract_from_dataset(paths["train_dir"], cache_tag="train")
    X_val, y_val = extractor.extract_from_dataset(paths["val_dir"], cache_tag="val")
    X_test, y_test = extractor.extract_from_dataset(paths["test_dir"], cache_tag="test")

    return X_train, y_train, X_val, y_val, X_test, y_test
