"""Export trained YOLOv8 model to various formats (ONNX, TorchScript, TFLite).

Usage:
    python scripts/export_detection.py --model results/detection/yolov8_disease/weights/best.pt --format onnx
    python scripts/export_detection.py --format torchscript
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to various formats")
    parser.add_argument("--model", type=str, default="results/detection/yolov8_disease/weights/best.pt", help="Path to YOLO model")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "tflite", "tf_lite", "saved_model", "engine", "coreml"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--half", action="store_true", help="Export with FP16 precision")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (ONNX only)")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace size in GB")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("MODEL EXPORT")
    print("=" * 60)
    print(f"  Input model: {model_path}")
    print(f"  Format: {args.format}")
    print(f"  Image size: {args.imgsz}")
    if args.format == "onnx":
        print(f"  Opset: {args.opset}")
        print(f"  Simplify: {args.simplify}")
        print(f"  Dynamic shapes: {args.dynamic}")
    print("=" * 60)

    model = YOLO(str(model_path))

    export_kwargs = {
        "imgsz": args.imgsz,
        "verbose": True,
    }

    if args.format in ["onnx", "engine"]:
        export_kwargs["opset"] = args.opset
        if args.format == "onnx":
            export_kwargs["simplify"] = args.simplify
            export_kwargs["dynamic"] = args.dynamic

    if args.format == "torchscript":
        export_kwargs["half"] = args.half

    if args.format == "tflite":
        export_kwargs["int8"] = False
        export_kwargs["half"] = args.half

    if args.format == "engine":
        export_kwargs["half"] = args.half
        export_kwargs["workspace"] = args.workspace

    print("\nExporting...")
    exported_path = model.export(format=args.format, **export_kwargs)

    print(f"\n{'='*60}")
    print(f"Export successful!")
    print(f"  Exported model: {exported_path}")
    print(f"{'='*60}")

    export_p = Path(exported_path)
    if export_p.exists():
        size_mb = export_p.stat().st_size / (1024 * 1024)
        print(f"\n  File size: {size_mb:.2f} MB")

    print("\nTo use the exported model:")
    if args.format == "onnx":
        print("  from ultralytics import YOLO")
        print("  model = YOLO('model.onnx')")
        print("  results = model('image.jpg')")
    elif args.format == "torchscript":
        print("  import torch")
        print("  model = torch.jit.load('model.torchscript')")
    elif args.format in ["tflite", "tf_lite"]:
        print("  # TensorFlow Lite")
        print("  import tensorflow as tf")
        print("  interpreter = tf.lite.Interpreter(model_path='model.tflite')")

    return exported_path


if __name__ == "__main__":
    main()
