"""
MobileNetV3 model implementation with fine-tuning support.
Uses inverted residuals and h-swish activation.
"""

from typing import Literal, Optional, List

import torch
import torch.nn as nn
import torchvision.models as models


MobileNetV3Variant = Literal["mobilenetv3_large", "mobilenetv3_small"]


def _load_mobilenetv3_base(name: str, pretrained: bool) -> nn.Module:
    """Load base MobileNetV3 model from torchvision."""
    if name == "mobilenetv3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v3_large(weights=weights)
    elif name == "mobilenetv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Unsupported MobileNetV3 variant: {name}")


def _freeze_backbone(model: nn.Module, unfreeze_from_layer: int = 12) -> None:
    """Freeze backbone layers, optionally unfreezing from a specific layer.

    MobileNetV3-Large has 17 InvertedResidual blocks (0-16) in features.
    MobileNetV3-Small has 13 blocks (0-12).

    Args:
        model: MobileNetV3 model
        unfreeze_from_layer: Unfreeze layers from this index onwards.
                            Set to -1 to freeze all backbone, 0 to unfreeze all.
    """
    # First, freeze everything in features
    for param in model.features.parameters():
        param.requires_grad = False

    # Then unfreeze from specified layer onwards
    if unfreeze_from_layer >= 0:
        num_layers = len(model.features)
        for layer_idx in range(unfreeze_from_layer, num_layers):
            for param in model.features[layer_idx].parameters():
                param.requires_grad = True


def create_mobilenetv3(
    name: str = "mobilenetv3_large",
    num_classes: int = 5,
    dropout: float = 0.2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_from_layer: int = 12,
) -> nn.Module:
    """Create MobileNetV3 model with custom classifier for fine-tuning.

    Args:
        name: Model variant (mobilenetv3_large, mobilenetv3_small)
        num_classes: Number of output classes
        dropout: Dropout rate for classifier
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        unfreeze_from_layer: Layer index from which to unfreeze

    Returns:
        MobileNetV3 model with modified classifier

    Example:
        >>> model = create_mobilenetv3("mobilenetv3_large", num_classes=5)
        >>> model = create_mobilenetv3("mobilenetv3_large", freeze_backbone=True)
    """
    model = _load_mobilenetv3_base(name, pretrained)

    # Get the number of features
    # MobileNetV3-Large: 960 -> 1280 -> num_classes
    # MobileNetV3-Small: 576 -> 1024 -> num_classes
    in_features = model.classifier[0].in_features
    hidden_features = model.classifier[0].out_features

    # Replace classifier with custom head
    # Keep the original structure: Linear -> Hardswish -> Dropout -> Linear
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(hidden_features, num_classes),
    )

    # Apply fine-tuning strategy
    if freeze_backbone:
        _freeze_backbone(model, unfreeze_from_layer)

    return model


class MobileNetV3Large(nn.Module):
    """MobileNetV3-Large wrapper class with fine-tuning utilities.

    Provides a cleaner interface for:
    - Gradual unfreezing during training
    - Discriminative learning rates
    - Easy access to feature extraction layers

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to initially freeze backbone
        unfreeze_from_layer: Layer to start unfreezing from (0-16)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_from_layer: int = 12,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.freeze_backbone_flag = freeze_backbone
        self.unfreeze_from_layer = unfreeze_from_layer

        # Load base model
        base_model = _load_mobilenetv3_base("mobilenetv3_large", pretrained)

        # Feature extractor (backbone)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Get number of features
        in_features = base_model.classifier[0].in_features
        hidden_features = base_model.classifier[0].out_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

        # Apply initial freezing strategy
        if freeze_backbone:
            self._freeze_backbone(unfreeze_from_layer)

    def _freeze_backbone(self, unfreeze_from_layer: int = -1) -> None:
        """Freeze backbone, optionally unfreezing from a layer."""
        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_from_layer >= 0:
            num_layers = len(self.features)
            for layer_idx in range(unfreeze_from_layer, num_layers):
                for param in self.features[layer_idx].parameters():
                    param.requires_grad = True

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning.

        Args:
            from_layer: Start unfreezing from this layer. If None, unfreeze all.
        """
        if from_layer is None:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            num_layers = len(self.features)
            for layer_idx in range(from_layer, num_layers):
                for param in self.features[layer_idx].parameters():
                    param.requires_grad = True

    def freeze_all_but_classifier(self) -> None:
        """Freeze all layers except the classifier head."""
        self._freeze_backbone(-1)

    def get_layer_groups(self) -> List[list]:
        """Get layer groups for discriminative learning rates.

        Returns:
            List of [early_layers, middle_layers, late_layers, classifier]
        """
        num_layers = len(self.features)
        third = num_layers // 3

        early_layers = list(self.features[:third].parameters())
        middle_layers = list(self.features[third:2*third].parameters())
        late_layers = list(self.features[2*third:].parameters())
        classifier = list(self.classifier.parameters())

        return [early_layers, middle_layers, late_layers, classifier]

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


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small wrapper class with fine-tuning utilities.

    Smaller and faster version for mobile/edge deployment.

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to initially freeze backbone
        unfreeze_from_layer: Layer to start unfreezing from (0-12)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_from_layer: int = 9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Load base model
        base_model = _load_mobilenetv3_base("mobilenetv3_small", pretrained)

        # Feature extractor
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Get number of features
        in_features = base_model.classifier[0].in_features
        hidden_features = base_model.classifier[0].out_features

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone(unfreeze_from_layer)

    def _freeze_backbone(self, unfreeze_from_layer: int = -1) -> None:
        """Freeze backbone layers."""
        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_from_layer >= 0:
            num_layers = len(self.features)
            for layer_idx in range(unfreeze_from_layer, num_layers):
                for param in self.features[layer_idx].parameters():
                    param.requires_grad = True

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze backbone layers."""
        if from_layer is None:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            num_layers = len(self.features)
            for layer_idx in range(from_layer, num_layers):
                for param in self.features[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_target_layer_for_gradcam(self):
        """Get the target layer for Grad-CAM visualization."""
        return self.features[-1]


def get_mobilenetv3_info(name: str = "mobilenetv3_large") -> dict:
    """Get information about MobileNetV3 architecture.

    Args:
        name: Model variant

    Returns:
        Dictionary with model info
    """
    info = {
        "mobilenetv3_large": {
            "input_size": 224,
            "params_millions": 5.4,
            "top1_imagenet": 75.2,
            "features_out": 960,
            "hidden_features": 1280,
            "num_layers": 17,
        },
        "mobilenetv3_small": {
            "input_size": 224,
            "params_millions": 2.5,
            "top1_imagenet": 67.4,
            "features_out": 576,
            "hidden_features": 1024,
            "num_layers": 13,
        },
    }
    return info.get(name, {})

