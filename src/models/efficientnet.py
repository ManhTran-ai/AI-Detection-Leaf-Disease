"""
EfficientNet-B0 model implementation with fine-tuning support.
Uses compound scaling and squeeze-excitation blocks.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.models as models


EfficientNetVariant = Literal["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]


def _load_efficientnet_base(name: str, pretrained: bool) -> nn.Module:
    """Load base EfficientNet model from torchvision."""
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights)
    elif name == "efficientnet_b1":
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b1(weights=weights)
    elif name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b2(weights=weights)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {name}")


def _freeze_backbone(model: nn.Module, unfreeze_from_block: int = 5) -> None:
    """Freeze backbone layers, optionally unfreezing from a specific block.

    EfficientNet-B0 has 9 blocks (0-8) in the features module.

    Args:
        model: EfficientNet model
        unfreeze_from_block: Unfreeze blocks from this index onwards (0-8).
                            Set to -1 to freeze all backbone, 0 to unfreeze all.
    """
    # First, freeze everything in features
    for param in model.features.parameters():
        param.requires_grad = False

    # Then unfreeze from specified block onwards
    if unfreeze_from_block >= 0:
        num_blocks = len(model.features)
        for block_idx in range(unfreeze_from_block, num_blocks):
            for param in model.features[block_idx].parameters():
                param.requires_grad = True


def create_efficientnet(
    name: str = "efficientnet_b0",
    num_classes: int = 5,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_from_block: int = 5,
) -> nn.Module:
    """Create EfficientNet model with custom classifier for fine-tuning.

    Args:
        name: Model variant (efficientnet_b0, efficientnet_b1, efficientnet_b2)
        num_classes: Number of output classes
        dropout: Dropout rate for classifier
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        unfreeze_from_block: Block index from which to unfreeze (0-8 for B0)

    Returns:
        EfficientNet model with modified classifier

    Example:
        >>> model = create_efficientnet("efficientnet_b0", num_classes=5, pretrained=True)
        >>> model = create_efficientnet("efficientnet_b0", freeze_backbone=True, unfreeze_from_block=6)
    """
    model = _load_efficientnet_base(name, pretrained)

    # Get the number of features from the last conv layer
    # EfficientNet-B0: 1280, B1: 1280, B2: 1408
    in_features = model.classifier[1].in_features

    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    # Apply fine-tuning strategy
    if freeze_backbone:
        _freeze_backbone(model, unfreeze_from_block)

    return model


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 wrapper class with fine-tuning utilities.

    Provides a cleaner interface for:
    - Gradual unfreezing during training
    - Discriminative learning rates
    - Easy access to feature extraction layers

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to initially freeze backbone
        unfreeze_from_block: Block to start unfreezing from (0-8)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_from_block: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.freeze_backbone = freeze_backbone
        self.unfreeze_from_block = unfreeze_from_block

        # Load base model
        base_model = _load_efficientnet_base("efficientnet_b0", pretrained)

        # Feature extractor (backbone)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Get number of features
        in_features = base_model.classifier[1].in_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # Apply initial freezing strategy
        if freeze_backbone:
            self._freeze_backbone(unfreeze_from_block)

    def _freeze_backbone(self, unfreeze_from_block: int = -1) -> None:
        """Freeze backbone, optionally unfreezing from a block."""
        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_from_block >= 0:
            num_blocks = len(self.features)
            for block_idx in range(unfreeze_from_block, num_blocks):
                for param in self.features[block_idx].parameters():
                    param.requires_grad = True

    def unfreeze_backbone(self, from_block: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning.

        Args:
            from_block: Start unfreezing from this block. If None, unfreeze all.
        """
        if from_block is None:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            num_blocks = len(self.features)
            for block_idx in range(from_block, num_blocks):
                for param in self.features[block_idx].parameters():
                    param.requires_grad = True

    def freeze_all_but_classifier(self) -> None:
        """Freeze all layers except the classifier head."""
        self._freeze_backbone(-1)

    def get_layer_groups(self) -> list:
        """Get layer groups for discriminative learning rates.

        Returns:
            List of [early_blocks, middle_blocks, late_blocks, classifier]
        """
        num_blocks = len(self.features)
        mid_point = num_blocks // 2

        early_blocks = list(self.features[:mid_point].parameters())
        middle_blocks = list(self.features[mid_point:-2].parameters())
        late_blocks = list(self.features[-2:].parameters())
        classifier = list(self.classifier.parameters())

        return [early_blocks, middle_blocks, late_blocks, classifier]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_target_layer_for_gradcam(self):
        """Get the target layer for Grad-CAM visualization.

        Returns the last convolutional block in features.
        """
        return self.features[-1]


def get_efficientnet_info(name: str = "efficientnet_b0") -> dict:
    """Get information about EfficientNet architecture.

    Args:
        name: Model variant

    Returns:
        Dictionary with model info including input size, params, etc.
    """
    info = {
        "efficientnet_b0": {
            "input_size": 224,
            "params_millions": 5.3,
            "top1_imagenet": 77.1,
            "features_out": 1280,
            "num_blocks": 9,
        },
        "efficientnet_b1": {
            "input_size": 240,
            "params_millions": 7.8,
            "top1_imagenet": 79.1,
            "features_out": 1280,
            "num_blocks": 9,
        },
        "efficientnet_b2": {
            "input_size": 260,
            "params_millions": 9.2,
            "top1_imagenet": 80.1,
            "features_out": 1408,
            "num_blocks": 9,
        },
    }
    return info.get(name, {})

