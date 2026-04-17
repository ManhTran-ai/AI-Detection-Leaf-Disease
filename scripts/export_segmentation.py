"""Export YOLOv8 Instance Segmentation model to various formats.

Usage:
    python scripts/export_segmentation.py --model results/segmentation/yolov8_seg_disease/weights/best.pt --format onnx
    python scripts/export_segmentation.py --model results/segmentation/yolov8_seg_disease/weights/best.pt --format torchscript
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


SUPPORTED_FORMATS = ["onnx", "torchscript", "tflite", "coreml", "saved_model", "pb", "tflite_float32", "edgetpu", "tfjs"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 Segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        default="results/segmentation/yolov8_seg_disease/weights/best.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=SUPPORTED_FORMATS,
        help="Export format",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 precision",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("YOLOv8 SEGMENTATION MODEL EXPORT")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Format: {args.format}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Half precision: {args.half}")
    print("=" * 60)

    model = YOLO(str(model_path))

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
    }

    if args.format == "onnx":
        export_kwargs["opset"] = args.opset
        export_kwargs["simplify"] = args.simplify

    try:
        exported_path = model.export(**export_kwargs)
        print(f"\n{'='*60}")
        print(f"EXPORT SUCCESSFUL")
        print(f"{'='*60}")
        print(f"  Saved to: {exported_path}")

        exported_path = Path(exported_path)
        file_size_mb = exported_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\nEXPORT FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
