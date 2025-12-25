import torch

from src.models.custom_cnn import CustomCNN
from src.models.resnet import create_resnet
from src.models.efficientnet import create_efficientnet, EfficientNetB0
from src.models.mobilenet import create_mobilenetv3, MobileNetV3Large


def test_resnet_output_shape():
    model = create_resnet("resnet18", num_classes=4, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 4)


def test_custom_cnn_output_shape():
    model = CustomCNN(in_channels=3, num_classes=4)
    dummy = torch.randn(2, 3, 128, 128)
    outputs = model(dummy)
    assert outputs.shape == (2, 4)


def test_efficientnet_b0_output_shape():
    """Test EfficientNet-B0 model output shape."""
    model = create_efficientnet("efficientnet_b0", num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 5)


def test_efficientnet_b0_wrapper_output_shape():
    """Test EfficientNetB0 wrapper class output shape."""
    model = EfficientNetB0(num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 5)


def test_efficientnet_freeze_backbone():
    """Test EfficientNet backbone freezing functionality."""
    model = EfficientNetB0(num_classes=5, pretrained=False, freeze_backbone=True, unfreeze_from_block=7)

    # Check that early blocks are frozen
    for param in model.features[0].parameters():
        assert not param.requires_grad

    # Check that classifier is trainable
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_efficientnet_gradcam_target():
    """Test that Grad-CAM target layer is accessible."""
    model = EfficientNetB0(num_classes=5, pretrained=False)
    target_layer = model.get_target_layer_for_gradcam()
    assert target_layer is not None


def test_mobilenetv3_large_output_shape():
    """Test MobileNetV3-Large model output shape."""
    model = create_mobilenetv3("mobilenetv3_large", num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 5)


def test_mobilenetv3_wrapper_output_shape():
    """Test MobileNetV3Large wrapper class output shape."""
    model = MobileNetV3Large(num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 5)


def test_mobilenetv3_freeze_backbone():
    """Test MobileNetV3 backbone freezing functionality."""
    model = MobileNetV3Large(num_classes=5, pretrained=False, freeze_backbone=True, unfreeze_from_layer=14)

    # Check that early layers are frozen
    for param in model.features[0].parameters():
        assert not param.requires_grad

    # Check that classifier is trainable
    for param in model.classifier.parameters():
        assert param.requires_grad


def test_mobilenetv3_gradcam_target():
    """Test that Grad-CAM target layer is accessible."""
    model = MobileNetV3Large(num_classes=5, pretrained=False)
    target_layer = model.get_target_layer_for_gradcam()
    assert target_layer is not None


def test_mobilenetv3_small_output_shape():
    """Test MobileNetV3-Small model output shape."""
    model = create_mobilenetv3("mobilenetv3_small", num_classes=5, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    assert outputs.shape == (2, 5)


