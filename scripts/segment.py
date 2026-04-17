"""Single image Instance Segmentation inference with YOLOv8.

Usage:
    python scripts/segment.py --model results/segmentation/yolov8_seg_disease/weights/best.pt --image path/to/image.jpg
    python scripts/segment.py --image path/to/image.jpg --conf 0.3 --save
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.utils import CLASS_NAMES, DISEASE_INFO_VN


SEGMENTATION_COLORS = {
    0: (0, 200, 0),       # ALGAL_LEAF_SPOT - Green
    1: (255, 165, 0),     # ALLOCARIDARA_ATTACK - Orange
    2: (0, 255, 0),       # HEALTHY_LEAF - Bright Green
    3: (255, 0, 0),       # LEAF_BLIGHT - Red
    4: (200, 0, 200),     # PHOMOPSIS_LEAF_SPOT - Purple
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instance Segmentation inference with YOLOv8"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/segmentation/yolov8_seg_disease/weights/best.pt",
        help="Path to YOLO segmentation model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for annotated image",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated image (non-blocking)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (0, 1, cpu)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Mask transparency (0-1)",
    )
    return parser.parse_args()


def draw_segmentation(
    image: np.ndarray,
    results,
    class_names: list,
    alpha: float = 0.4,
) -> tuple:
    """Draw segmentation masks on image.

    Args:
        image: Image array (H, W, C) in RGB format.
        results: YOLO results object.
        class_names: List of class names.
        alpha: Mask transparency (0-1).

    Returns:
        Tuple of (annotated_image, detections_list).
    """
    img = image.copy()
    h, w = img.shape[:2]

    detections = []
    boxes = results.boxes
    masks = results.masks

    if masks is None:
        return img, []

    polygons = masks.xy
    classes = masks.cls
    confidences = masks.conf

    for i in range(len(polygons)):
        cls_id = int(classes[i])
        conf = float(confidences[i])
        polygon = polygons[i]

        if cls_id >= len(class_names):
            continue

        cls_name = class_names[cls_id]
        color = SEGMENTATION_COLORS.get(cls_id, (255, 255, 255))

        pts = polygon.astype(np.int32).reshape((-1, 1, 2))

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (*color, int(255 * alpha)))
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.polylines(img, [pts], True, color, 2)

        center_x = int(np.mean(polygon[:, 0]))
        center_y = int(np.mean(polygon[:, 1]))
        label = f"{cls_name}: {conf:.2f}"

        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(center_y - 10, label_h + baseline + 10)
        label_x = max(center_x - label_w // 2, 5)

        cv2.rectangle(
            img,
            (label_x, label_y - label_h - baseline - 4),
            (label_x + label_w, label_y),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (label_x, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        mask_area = cv2.contourArea(pts)
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "polygon": polygon.tolist(),
            "mask_area_pixels": float(mask_area),
            "bbox": boxes.xyxy[i].tolist() if boxes is not None else None,
        })

    return img, detections


def print_predictions(predictions: list, class_names: list, disease_info: dict):
    """Print predictions in a formatted way."""
    if not predictions:
        print("  No segmentations found.")
        return

    print(f"\n  Found {len(predictions)} segmentation(s):")
    print(f"  {'#':<4} {'Class':<25} {'Confidence':>12} {'Mask Area':>12}")
    print("  " + "-" * 60)

    total_area = 0
    for i, pred in enumerate(predictions, 1):
        cls_id = pred["class_id"]
        conf = pred["confidence"]
        mask_area = pred["mask_area_pixels"]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"

        print(f"  {i:<4} {cls_name:<25} {conf:>11.2%} {mask_area:>10.0f} px")
        total_area += mask_area

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

    print(f"\n  Total affected area: {total_area:,.0f} pixels")
    print()


def main():
    args = parse_args()

    from ultralytics import YOLO

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train_segmentation.py --model yolov8n-seg --data data/seg_yolo/dataset.yaml")
        sys.exit(1)

    print("=" * 60)
    print("DURIAN DISEASE INSTANCE SEGMENTATION - INFERENCE")
    print("=" * 60)
    print(f"  Image: {image_path.name}")
    print(f"  Model: {model_path.name}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Mask alpha: {args.alpha}")
    print("=" * 60)

    device = args.device if args.device else ("0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    model = YOLO(str(model_path))

    results = model(
        str(image_path),
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )

    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    annotated_img, detections = draw_segmentation(
        img_rgb, results[0], CLASS_NAMES, alpha=args.alpha
    )

    print_predictions(detections, CLASS_NAMES, DISEASE_INFO_VN)

    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    if args.save and annotated_bgr is not None:
        output_dir = Path(args.output) if args.output else image_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{image_path.stem}_segmented{image_path.suffix}"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), annotated_bgr)
        print(f"\nAnnotated image saved to: {output_path}")

    if args.show and annotated_bgr is not None:
        cv2.imshow("Durian Disease Segmentation", annotated_bgr)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not detections:
        print("\nNo disease segments detected above the confidence threshold.")
        print("Try lowering --conf threshold.")

    return {"detections": detections, "annotated_image": annotated_bgr}


if __name__ == "__main__":
    main()
