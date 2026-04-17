"""Train YOLOv8 object detection model for durian disease detection.

Usage:
    python scripts/train_detection.py --model yolov8n --data data/detect_yolo/dataset.yaml
    python scripts/train_detection.py --model yolov8s --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_device, set_global_seed


AVAILABLE_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
}

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for durian disease detection")
    parser.add_argument("--model", type=str, default="yolov8n", choices=list(AVAILABLE_MODELS.keys()), help="YOLOv8 model variant")
    parser.add_argument("--data", type=str, default="data/detect_yolo/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (0, 1, cpu)")
    parser.add_argument("--project", type=str, default="results/detection", help="Project directory")
    parser.add_argument("--name", type=str, default="yolov8_disease", help="Experiment name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--save_period", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (SGD, Adam, AdamW)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for val")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--no_aug", action="store_true", help="Disable mosaic/mixup augmentation")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DURIAN DISEASE DETECTION - YOLOv8 TRAINING")
    print("=" * 60)

    set_global_seed(args.seed)

    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {device}")
    print("=" * 60)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"\nERROR: Dataset YAML not found: {data_yaml}")
        print("Please create the dataset first using:")
        print("  1. Annotate images with bounding boxes (COCO/VOC format)")
        print("  2. Convert to YOLO format using:")
        print("     python scripts/convert_annotations.py --format coco --input your_annotations.json")
        print("  Or create dataset.yaml manually at:", data_yaml)
        sys.exit(1)

    model_name = AVAILABLE_MODELS[args.model]
    model = YOLO(model_name)

    augmentation_config = {}
    if args.no_aug:
        augmentation_config = {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}
    else:
        augmentation_config = {
            "mosaic": 1.0,
            "mixup": 0.1,
            "degrees": 15.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "flipup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        }

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        save=True,
        save_period=args.save_period,
        patience=args.patience,
        verbose=args.verbose,
        amp=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        close_mosaic=10,
        **augmentation_config,
    )

    best_model = project_root / args.project / args.name / "weights" / "best.pt"
    if best_model.exists():
        print(f"\n{'='*60}")
        print(f"Training complete! Best model saved at:")
        print(f"  {best_model}")
        print(f"{'='*60}")

        print("\nEvaluating best model on validation set...")
        val_metrics = model.val(data=str(data_yaml), split="val", verbose=True)
        print(f"  mAP@0.50: {val_metrics.box.map50:.4f}")
        print(f"  mAP@0.50:0.95: {val_metrics.box.map:.4f}")
        print(f"  Precision: {val_metrics.box.mp:.4f}")
        print(f"  Recall: {val_metrics.box.mr:.4f}")
    else:
        print("\nWARNING: Best model not found. Check training output.")

    print("\nTo export the model:")
    print("  ONNX:       python scripts/export_detection.py --format onnx")
    print("  TorchScript: python scripts/export_detection.py --format torchscript")
    print("  TFLite:     python scripts/export_detection.py --format tflite")


if __name__ == "__main__":
    main()
