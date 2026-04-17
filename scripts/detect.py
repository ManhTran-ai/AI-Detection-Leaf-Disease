"""Single image/object detection inference with YOLOv8.

Usage:
    python scripts/detect.py --model results/detection/yolov8_disease/weights/best.pt --image path/to/image.jpg
    python scripts/detect.py --image path/to/image.jpg --conf 0.3 --save
"""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

import cv2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.detector import DetectionPredictor
from src.detection.utils import CLASS_NAMES, DISEASE_INFO_VN


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection inference with YOLOv8")
    parser.add_argument("--model", type=str, default="results/detection/yolov8_disease/weights/best.pt", help="Path to YOLO model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output directory for annotated image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true", help="Save annotated image")
    parser.add_argument("--show", action="store_true", help="Display annotated image (non-blocking)")
    parser.add_argument("--device", type=str, default=None, help="Device (0, 1, cpu)")
    return parser.parse_args()


def print_predictions(predictions: list, class_names: list, disease_info: dict):
    """Print predictions in a formatted way."""
    if not predictions:
        print("  No detections found.")
        return

    print(f"\n  Found {len(predictions)} detection(s):")
    print(f"  {'#':<4} {'Class':<25} {'Confidence':>12} {'BBox':>30}")
    print("  " + "-" * 75)

    for i, pred in enumerate(predictions, 1):
        cls_id = pred["class_id"]
        conf = pred["confidence"]
        bbox = pred["bbox"]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"

        bbox_str = f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
        print(f"  {i:<4} {cls_name:<25} {conf:>11.2%} {bbox_str:>30}")

        info = disease_info.get(cls_name, {})
        if info:
            vn_name = info.get("name", "")
            desc = info.get("description", "")
            treatment = info.get("treatment", "")
            print(f"       -> {vn_name}")
            if desc:
                print(f"          {desc[:100]}...")
            if treatment:
                print(f"          Treatment: {treatment[:100]}...")

    print()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_detection.py --model yolov8n --data data/detect_yolo/dataset.yaml")
        sys.exit(1)

    print("=" * 60)
    print("DURIAN DISEASE DETECTION - INFERENCE")
    print("=" * 60)
    print(f"  Image: {image_path.name}")
    print(f"  Model: {model_path.name}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print("=" * 60)

    predictor = DetectionPredictor(
        model_path=str(model_path),
        class_names=CLASS_NAMES,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    result = predictor.predict(str(image_path), return_image=True)

    print_predictions(result["predictions"], CLASS_NAMES, DISEASE_INFO_VN)

    if result["annotated_image"] is not None and args.save:
        output_dir = Path(args.output) if args.output else image_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{image_path.stem}_detected{image_path.suffix}"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), result["annotated_image"])
        print(f"\nAnnotated image saved to: {output_path}")

    if result["annotated_image"] is not None and args.show:
        cv2.imshow("Durian Disease Detection", result["annotated_image"])
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not result["predictions"]:
        print("\nNo disease detected above the confidence threshold.")
        print("Try lowering --conf threshold.")

    return result


if __name__ == "__main__":
    main()
