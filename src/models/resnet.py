from typing import Literal

import torch.nn as nn
import torchvision.models as models


ResNetVariant = Literal["resnet18", "resnet34", "resnet50"]


def _load_base_model(name: ResNetVariant, pretrained: bool):
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if name == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    raise ValueError(f"Unsupported ResNet variant: {name}")


def create_resnet(name: ResNetVariant, num_classes: int, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    model = _load_base_model(name, pretrained)
    in_features = model.fc.in_features
    classifier_layers = [nn.Dropout(dropout), nn.Linear(in_features, num_classes)]
    model.fc = nn.Sequential(*classifier_layers)
    return model


def get_resnet_feature_dim(name: ResNetVariant) -> int:
    """Return the feature dimension (in_features of fc layer) for a ResNet variant."""
    feat_dims = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
    return feat_dims.get(name, 512)


def replace_resnet_classifier_with_identity(model: nn.Module) -> nn.Module:
    """Remove the classifier head from a ResNet model for feature extraction.

    After calling this, forward pass returns features from GlobalAveragePooling
    instead of class logits. The feature dimension matches get_resnet_feature_dim().
    """
    model.fc = nn.Identity()
    return model


