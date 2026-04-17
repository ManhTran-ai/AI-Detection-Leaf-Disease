"""Batch annotation helper - semi-automated annotation tool for bounding boxes.

This script helps create COCO-format annotations by providing a simple UI
for drawing bounding boxes on images. For large-scale annotation, consider
using dedicated tools like CVAT, LabelImg, LabelMe, or Roboflow.

Usage:
    # Create COCO annotations from images
    python scripts/annotate_data.py --input data/raw_detect --output data/annotations/coco_annotations.json

    # Convert existing COCO/VOC to YOLO format
    python scripts/annotate_data.py --convert --input data/annotations/coco_annotations.json --output data/detect_yolo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.converter import coco_to_yolo, voc_to_yolo, create_dataset_from_class_folders
from src.detection.utils import CLASS_NAMES

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class SimpleAnnotator:
    """Simple bounding box annotator using OpenCV window.

    Controls:
        Mouse drag: Draw bounding box
        ESC: Exit / Next image
        SPACE: Save annotations and go to next
        R: Remove last box
        S: Save and skip to next image
        Q: Quit without saving
    """

    def __init__(self, image_path: str, class_names: List[str]):
        self.image_path = Path(image_path)
        self.class_names = class_names
        self.img = cv2.imread(str(image_path))
        if self.img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        self.clone = self.img.copy()
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes: List[Tuple[int, int, int, int, int]] = []
        self.current_class = 0
        self.window_name = f"Annotator - {self.image_path.name}"

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                preview = self.clone.copy()
                cv2.rectangle(preview, self.start_point, self.end_point, (0, 255, 0), 2)
                self._draw_all_boxes(preview)
                cv2.imshow(self.window_name, preview)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.boxes.append((self.current_class, x1, y1, x2, y2))
            self.start_point = None
            self.end_point = None

    def _draw_all_boxes(self, img: np.ndarray):
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (200, 0, 200)]
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = colors[cls % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = self.class_names[cls]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def annotate(self) -> Tuple[List[Tuple[int, int, int, int, int]], bool]:
        """Run annotation UI.

        Returns:
            (list of (class_id, x1, y1, x2, y2), should_save)
        """
        while True:
            display = self.clone.copy()
            self._draw_all_boxes(display)

            info = f"Class [{self.current_class}]: {self.class_names[self.current_class]} | Boxes: {len(self.boxes)}"
            info += " | SPACE=Save Next | R=Remove Last | Q=Quit | S=Skip | 0-4=Change Class"
            cv2.putText(display, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                return self.boxes, False
            elif key == ord(" "):
                cv2.destroyAllWindows()
                return self.boxes, True
            elif key == ord("s"):
                cv2.destroyAllWindows()
                return self.boxes, True
            elif key == ord("r"):
                if self.boxes:
                    self.boxes.pop()
            elif key in [ord(str(i)) for i in range(len(self.class_names))]:
                self.current_class = int(chr(key))


def coco_to_rects(coco_annotations: List[Dict], image_id: int) -> List[Tuple[int, int, int, int, int]]:
    """Convert COCO annotations to (class, x1, y1, x2, y2) format for a single image."""
    img_w = 0
    img_h = 0
    rects = []
    for ann in coco_annotations:
        if ann["image_id"] == image_id:
            img_h = ann.get("height", 0)
            img_w = ann.get("width", 0)
            x, y, w, h = ann["bbox"]
            rects.append((ann["category_id"], int(x), int(y), int(x + w), int(y + h)))
    return rects


def parse_args():
    parser = argparse.ArgumentParser(description="Annotation helper for durian disease detection")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    annotate_parser = subparsers.add_parser("annotate", help="Interactive annotation tool")
    annotate_parser.add_argument("--input", type=str, required=True, help="Input image directory")
    annotate_parser.add_argument("--output", type=str, required=True, help="Output COCO JSON path")
    annotate_parser.add_argument("--start_idx", type=int, default=0, help="Start index")

    convert_parser = subparsers.add_parser("convert", help="Convert COCO/VOC to YOLO")
    convert_parser.add_argument("--format", type=str, default="coco", choices=["coco", "voc", "class_folder"], help="Input format")
    convert_parser.add_argument("--input", type=str, required=True, help="Input file/directory")
    convert_parser.add_argument("--images", type=str, default=None, help="Images directory (for COCO)")
    convert_parser.add_argument("--output", type=str, required=True, help="Output YOLO dataset directory")
    convert_parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio")
    convert_parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio")
    convert_parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    convert_parser.add_argument("--copy_images", action="store_true", default=True, help="Copy images to output")

    stats_parser = subparsers.add_parser("stats", help="Show annotation statistics")
    stats_parser.add_argument("--input", type=str, required=True, help="COCO JSON file")
    stats_parser.add_argument("--images_dir", type=str, required=True, help="Images directory")

    return parser.parse_args()


def run_annotation(args):
    """Run interactive annotation."""
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to annotate")
    print("Controls: SPACE=Save&Next | R=Remove Last | 0-4=Change Class | Q=Quit")
    print("-" * 40)

    all_annotations = []
    next_image_id = args.start_idx

    for i, img_path in enumerate(tqdm(image_files[args.start_idx:], desc="Annotating")):
        try:
            annotator = SimpleAnnotator(str(img_path), CLASS_NAMES)
            boxes, should_save = annotator.annotate()

            if should_save and boxes:
                img_h, img_w = annotator.img.shape[:2]
                img_id = next_image_id

                for cls, x1, y1, x2, y2 in boxes:
                    all_annotations.append({
                        "id": len(all_annotations) + 1,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                    })

                next_image_id += 1
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_annotations, f, indent=2, ensure_ascii=False)

        except KeyboardInterrupt:
            print("\nAnnotation interrupted.")
            break

    print(f"\nSaved {len(all_annotations)} annotations to {output_path}")


def run_stats(args):
    """Show annotation statistics."""
    with open(args.input, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    from collections import Counter
    cls_counts = Counter(ann["category_id"] for ann in annotations)
    print("\nAnnotation Statistics:")
    print(f"  Total annotations: {len(annotations)}")
    print("  Per class:")
    for cls_id, count in sorted(cls_counts.items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
        print(f"    {name}: {count}")


def run_conversion(args):
    """Convert COCO/VOC to YOLO format."""
    if args.format == "coco":
        if not args.images:
            print("Error: --images directory required for COCO format")
            sys.exit(1)
        result = coco_to_yolo(
            coco_json_path=args.input,
            output_dir=args.output,
            images_dir=args.images,
            class_names=CLASS_NAMES,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            copy_images=args.copy_images,
        )
    elif args.format == "voc":
        result = voc_to_yolo(
            voc_annotations_dir=args.input,
            voc_images_dir=args.images or args.input,
            output_dir=args.output,
            class_names=CLASS_NAMES,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    elif args.format == "class_folder":
        result = create_dataset_from_class_folders(
            source_dir=args.input,
            output_dir=args.output,
            class_names=CLASS_NAMES,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    print(f"\nConversion complete!")
    print(f"  Dataset YAML: {result.get('dataset_yaml')}")
    print(f"  Train: {result.get('num_train')} images")
    print(f"  Val: {result.get('num_val')} images")
    print(f"  Test: {result.get('num_test')} images")


def main():
    args = parse_args()
    if args.command == "annotate":
        run_annotation(args)
    elif args.command == "stats":
        run_stats(args)
    elif args.command == "convert":
        run_conversion(args)
    else:
        print("Usage:")
        print("  Annotate: python scripts/annotate_data.py annotate --input DIR --output JSON")
        print("  Convert:  python scripts/annotate_data.py convert --format coco --input JSON --images DIR --output YOLO_DIR")
        print("  Stats:    python scripts/annotate_data.py stats --input JSON --images_dir DIR")


if __name__ == "__main__":
    main()
