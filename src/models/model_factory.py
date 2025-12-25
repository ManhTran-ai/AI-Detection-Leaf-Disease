from typing import Dict

import torch.nn as nn

from .custom_cnn import CustomCNN
from .resnet import create_resnet
from .efficientnet import create_efficientnet
from .mobilenet import create_mobilenetv3


def build_model(config: Dict) -> nn.Module:
    """Build a model based on configuration.

    Supported models:
    - ResNet: resnet18, resnet34, resnet50
    - EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
    - MobileNetV3: mobilenetv3_large, mobilenetv3_small
    - CustomCNN: custom_cnn

    Args:
        config: Configuration dictionary containing model and dataset settings

    Returns:
        PyTorch model instance
    """
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    num_classes = len(dataset_cfg["class_names"])
    name = model_cfg["name"].lower()

    # ResNet variants
    if name in {"resnet18", "resnet34", "resnet50"}:
        return create_resnet(
            name=name,
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.3),
            pretrained=model_cfg.get("pretrained", True),
        )

    # EfficientNet variants
    if name in {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2"}:
        return create_efficientnet(
            name=name,
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.3),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            unfreeze_from_block=model_cfg.get("unfreeze_from_block", 5),
        )

    # MobileNetV3 variants
    if name in {"mobilenetv3_large", "mobilenetv3_small"}:
        return create_mobilenetv3(
            name=name,
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            unfreeze_from_layer=model_cfg.get("unfreeze_from_layer", 12),
        )

    # Custom CNN
    if name == "custom_cnn":
        return CustomCNN(
            in_channels=model_cfg.get("in_channels", 3),
            num_classes=num_classes,
            conv_filters=model_cfg.get("conv_filters"),
            dropout=model_cfg.get("dropout", 0.4),
            fc_units=model_cfg.get("fc_units", 512),
        )

    raise ValueError(f"Unsupported model name: {name}. "
                    f"Supported: resnet18, resnet34, resnet50, "
                    f"efficientnet_b0, efficientnet_b1, efficientnet_b2, "
                    f"mobilenetv3_large, mobilenetv3_small, custom_cnn")


