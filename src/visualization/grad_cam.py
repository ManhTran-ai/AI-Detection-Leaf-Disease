from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image


def get_target_layer_for_model(model: torch.nn.Module, model_name: Optional[str] = None):
    """Get the appropriate target layer for Grad-CAM based on model architecture.

    Args:
        model: PyTorch model
        model_name: Optional model name hint (resnet18, efficientnet_b0, etc.)

    Returns:
        Target layer for Grad-CAM visualization
    """
    # Try to detect model type from model name if provided
    if model_name:
        model_name = model_name.lower()

        # ResNet family
        if "resnet" in model_name:
            return model.layer4[-1]  # Last block of layer4

        # EfficientNet family
        if "efficientnet" in model_name:
            return model.features[-1]  # Last MBConv block

        # MobileNetV3 family
        if "mobilenet" in model_name:
            return model.features[-1]  # Last InvertedResidual block

    # Auto-detect from model architecture
    if hasattr(model, 'layer4'):  # ResNet-like
        return model.layer4[-1]
    elif hasattr(model, 'features'):  # EfficientNet/MobileNet-like
        return model.features[-1]
    elif hasattr(model, 'get_target_layer_for_gradcam'):  # Custom method
        return model.get_target_layer_for_gradcam()

    raise ValueError("Could not automatically detect target layer for Grad-CAM. "
                    "Please provide the target layer manually.")


def generate_grad_cam(
    model: torch.nn.Module,
    target_layer: Union[torch.nn.Module, str],
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: Optional[int],
    output_path: str,
    model_name: Optional[str] = None,
) -> str:
    """Generate Grad-CAM visualization for a model prediction.

    Args:
        model: PyTorch model
        target_layer: Target layer for Grad-CAM (or "auto" for automatic detection)
        input_tensor: Preprocessed input tensor
        original_image: Original image as numpy array (RGB)
        class_idx: Target class index (None for predicted class)
        output_path: Path to save the output visualization
        model_name: Optional model name for automatic target layer detection

    Returns:
        Path to saved visualization
    """
    model.eval()

    # Auto-detect target layer if needed
    if target_layer == "auto" or target_layer is None:
        target_layer = get_target_layer_for_model(model, model_name)

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_backward_hook(backward_hook)

    input_batch = input_tensor.unsqueeze(0).to(next(model.parameters()).device)
    output = model(input_batch)
    if class_idx is None:
        class_idx = int(output.argmax(dim=1).item())
    target = output[0, class_idx]
    model.zero_grad()
    target.backward()

    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze(0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        heatmap,
        0.4,
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
        0.6,
        0,
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)

    f_handle.remove()
    b_handle.remove()
    return output_path

