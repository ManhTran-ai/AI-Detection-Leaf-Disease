"""Evaluate YOLOv8 Instance Segmentation model on test set.

Usage:
    python scripts/evaluate_segmentation.py --model results/segmentation/yolov8_seg_disease/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.utils import CLASS_NAMES


CLASS_NAMES_SEGMENTATION = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 Instance Segmentation model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO segmentation model (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/seg_yolo/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="Save results to TXT",
    )
    parser.add_argument(
        "--save_seg",
        action="store_true",
        help="Save segmentation predictions",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="results/segmentation",
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="eval",
        help="Experiment name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DURIAN DISEASE INSTANCE SEGMENTATION - MODEL EVALUATION")
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
        save=args.save_seg,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if hasattr(metrics, "box") and metrics.box is not None:
        print(f"\n[Detection Metrics - Bounding Box]")
        print(f"  {'Metric':<25} {'Value':>10}")
        print(f"  {'-'*37}")
        print(f"  {'mAP@0.50':<25} {metrics.box.map50:>10.4f}")
        print(f"  {'mAP@0.50:0.95':<25} {metrics.box.map:>10.4f}")
        print(f"  {'Precision':<25} {metrics.box.mp:>10.4f}")
        print(f"  {'Recall':<25} {metrics.box.mr:>10.4f}")
        
        if metrics.box.mp > 0 and metrics.box.mr > 0:
            f1_box = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)
            print(f"  {'F1 Score (box)':<25} {f1_box:>10.4f}")

    if hasattr(metrics, "seg") and metrics.seg is not None:
        print(f"\n[Segmentation Metrics - Mask]")
        print(f"  {'Metric':<25} {'Value':>10}")
        print(f"  {'-'*37}")
        print(f"  {'mAP@0.50 (mask)':<25} {metrics.seg.map50:>10.4f}")
        print(f"  {'mAP@0.50:0.95 (mask)':<25} {metrics.seg.map:>10.4f}")
        print(f"  {'Precision (mask)':<25} {metrics.seg.mp:>10.4f}")
        print(f"  {'Recall (mask)':<25} {metrics.seg.mr:>10.4f}")
        
        if metrics.seg.mp > 0 and metrics.seg.mr > 0:
            f1_seg = 2 * metrics.seg.mp * metrics.seg.mr / (metrics.seg.mp + metrics.seg.mr + 1e-8)
            print(f"  {'F1 Score (mask)':<25} {f1_seg:>10.4f}")

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print(f"\n[Per-Class Metrics]")
        print(f"  {'Class':<25} {'AP50(box)':>10} {'AP50(seg)':>10} {'P(box)':>10} {'R(box)':>10}")
        print(f"  {'-'*70}")
        
        seg_ap50 = None
        if hasattr(metrics.seg, "ap50") and metrics.seg.ap50 is not None:
            seg_ap50 = metrics.seg.ap50
        
        for cls_idx in metrics.box.ap_class_index:
            if cls_idx < len(CLASS_NAMES_SEGMENTATION):
                cls_name = CLASS_NAMES_SEGMENTATION[cls_idx]
                ap50_box = float(metrics.box.ap50[cls_idx]) if hasattr(metrics.box, "ap50") else 0.0
                ap50_seg = float(seg_ap50[cls_idx]) if seg_ap50 is not None else 0.0
                p_box = float(metrics.box.p[cls_idx]) if hasattr(metrics.box, "p") else 0.0
                r_box = float(metrics.box.r[cls_idx]) if hasattr(metrics.box, "r") else 0.0
                
                print(f"  {cls_name:<25} {ap50_box:>10.4f} {ap50_seg:>10.4f} {p_box:>10.4f} {r_box:>10.4f}")

    print("=" * 60)

    eval_dir = Path(args.project) / args.name
    if args.save_json:
        json_path = eval_dir / "results.json"
        print(f"\nResults saved to: {json_path}")
    
    if args.save_seg:
        seg_dir = eval_dir / "segmentations"
        print(f"Segmentation predictions saved to: {seg_dir}")

    return metrics


if __name__ == "__main__":
    main()
