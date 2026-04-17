"""Evaluate YOLOv8 detection model on test set.

Usage:
    python scripts/evaluate_detection.py --model results/detection/yolov8_disease/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.metrics import DetectionMetrics
from src.detection.utils import CLASS_NAMES

CLASS_NAMES_DETECTION = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (.pt)")
    parser.add_argument("--data", type=str, default="data/detect_yolo/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate on")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--save_json", action="store_true", help="Save results to JSON")
    parser.add_argument("--save_txt", action="store_true", help="Save results to TXT")
    parser.add_argument("--project", type=str, default="results/detection", help="Project directory")
    parser.add_argument("--name", type=str, default="eval", help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DURIAN DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"\nERROR: Dataset YAML not found: {data_yaml}")
        sys.exit(1)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Split: {args.split}")
    print(f"Conf threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("=" * 60)

    model = YOLO(str(model_path))

    print(f"\nEvaluating on {args.split} set...")
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=args.save_json,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.50     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95: {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)
    print(f"  F1 Score     : {f1:.4f}")
    print("-" * 60)

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print("Per-Class Metrics:")
        print(f"  {'Class':<25} {'AP50':>8} {'AP':>8} {'P':>8} {'R':>8} {'F1':>8}")
        print("  " + "-" * 65)
        for cls_idx in metrics.box.ap_class_index:
            if cls_idx < len(CLASS_NAMES_DETECTION):
                cls_name = CLASS_NAMES_DETECTION[cls_idx]
                ap50 = float(metrics.box.ap50[cls_idx]) if hasattr(metrics.box, "ap50") else 0.0
                ap = float(metrics.box.ap[cls_idx]) if hasattr(metrics.box, "ap") else 0.0
                p = float(metrics.box.p[cls_idx]) if hasattr(metrics.box, "p") else 0.0
                r = float(metrics.box.r[cls_idx]) if hasattr(metrics.box, "r") else 0.0
                f1_cls = 2 * p * r / (p + r + 1e-8)
                print(f"  {cls_name:<25} {ap50:>8.4f} {ap:>8.4f} {p:>8.4f} {r:>8.4f} {f1_cls:>8.4f}")

    print("=" * 60)

    eval_dir = Path(args.project) / args.name
    if args.save_json:
        json_path = eval_dir / "results.json"
        print(f"\nResults saved to: {json_path}")

    return metrics


if __name__ == "__main__":
    main()
