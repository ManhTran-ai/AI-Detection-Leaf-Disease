import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import Predictor
from src.utils.config import get_device, load_config
from src.visualization.grad_cam import generate_grad_cam


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to image for prediction.")
    parser.add_argument("--output", default="outputs/predictions/prediction.json", help="Where to save prediction JSON.")
    parser.add_argument("--grad-cam", default=None, help="Optional path to save Grad-CAM visualization.")
    return parser.parse_args()


def _get_target_layer(model: nn.Module):
    if hasattr(model, "layer4"):
        block = model.layer4[-1]
        if hasattr(block, "conv3"):
            return block.conv3
        if hasattr(block, "conv2"):
            return block.conv2
        return block
    if hasattr(model, "features"):
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
    raise ValueError("Unable to determine target layer for Grad-CAM.")


def main():
    args = parse_args()
    config = load_config(args.config).raw
    device = get_device()

    predictor = Predictor(config=config, checkpoint_path=args.checkpoint, device=device)
    result = predictor.predict(args.image)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            },
            fp,
            indent=2,
        )

    if args.grad_cam:
        target_layer = _get_target_layer(predictor.model)
        generate_grad_cam(
            model=predictor.model,
            target_layer=target_layer,
            input_tensor=result["input_tensor"],
            original_image=result["image"],
            class_idx=None,
            output_path=args.grad_cam,
        )

    print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    main()

