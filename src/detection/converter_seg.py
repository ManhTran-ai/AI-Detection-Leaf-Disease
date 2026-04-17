"""Convert COCO Segmentation annotations to YOLO Segmentation format.

This converter handles COCO JSON files with polygon segmentations and converts
them to YOLO segmentation format (.txt files with normalized polygon coordinates).

Usage:
    python scripts/convert_to_segmentation.py --input annotations.json --images data/raw --output data/seg_yolo
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def coco_seg_to_yolo_seg(
    coco_json_path: str,
    output_dir: str,
    images_dir: str,
    class_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_images: bool = True,
) -> Dict[str, str]:
    """Convert COCO Segmentation annotations to YOLO Segmentation format.

    Args:
        coco_json_path: Path to COCO format JSON file with segmentation annotations.
        output_dir: Output directory for YOLO segmentation dataset.
        images_dir: Directory containing images.
        class_names: List of class names (must match COCO categories).
        train_ratio: Ratio of training images.
        val_ratio: Ratio of validation images.
        test_ratio: Ratio of test images.
        copy_images: If True, copy images to train/val/test folders.

    Returns:
        Dict with paths to dataset.yaml and split directories.
    """
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_map = {img["id"]: img for img in coco["images"]}
    categories_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations_by_image: Dict[int, List] = {img_id: [] for img_id in images_map}
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    category_to_idx = {name: idx for idx, name in enumerate(class_names)}

    splits = {
        "train": output_dir / "images" / "train",
        "val": output_dir / "images" / "val",
        "test": output_dir / "images" / "test",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
        "test_labels": output_dir / "labels" / "test",
    }

    for path in splits.values():
        path.mkdir(parents=True, exist_ok=True)

    image_ids = list(images_map.keys())
    np.random.seed(42)
    indices = np.random.permutation(len(image_ids))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_ids = set(image_ids[i] for i in indices[:n_train])
    val_ids = set(image_ids[i] for i in indices[n_train : n_train + n_val])
    test_ids = set(image_ids[i] for i in indices[n_train + n_val :])

    id_to_split = {}
    for img_id in train_ids:
        id_to_split[img_id] = "train"
    for img_id in val_ids:
        id_to_split[img_id] = "val"
    for img_id in test_ids:
        id_to_split[img_id] = "test"

    stats = {"images_processed": 0, "annotations_processed": 0, "skipped": 0}

    for img_id, img_info in tqdm(images_map.items(), desc="Converting segmentation annotations"):
        split = id_to_split[img_id]
        img_width = img_info["width"]
        img_height = img_info["height"]
        img_file_name = img_info["file_name"]
        img_path = images_dir / img_file_name

        label_file = output_dir / "labels" / split / (Path(img_file_name).stem + ".txt")
        with open(label_file, "w", encoding="utf-8") as f:
            for ann in annotations_by_image[img_id]:
                cat_name = categories_map[ann["category_id"]]
                if cat_name not in category_to_idx:
                    continue
                cls_idx = category_to_idx[cat_name]

                if "segmentation" in ann and ann["segmentation"]:
                    segmentation = ann["segmentation"][0]
                    normalized_coords = []

                    for i in range(0, len(segmentation), 2):
                        x = segmentation[i] / img_width
                        y = segmentation[i + 1] / img_height
                        normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])

                    coords_str = " ".join(normalized_coords)
                    f.write(f"{cls_idx} {coords_str}\n")
                    stats["annotations_processed"] += 1

        if copy_images and img_path.exists():
            dest_img = output_dir / "images" / split / img_file_name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            img_data = cv2.imread(str(img_path))
            if img_data is not None:
                cv2.imwrite(str(dest_img), img_data)
                stats["images_processed"] += 1
            else:
                stats["skipped"] += 1

    yaml_path = write_dataset_yaml(output_dir, class_names)

    print(f"\nConversion Summary:")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Annotations processed: {stats['annotations_processed']}")
    print(f"  Skipped (missing images): {stats['skipped']}")

    return {
        "dataset_yaml": str(yaml_path),
        "train_dir": str(splits["train"]),
        "val_dir": str(splits["val"]),
        "test_dir": str(splits["test"]),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
    }


def convert_existing_split(
    coco_json_path: str,
    output_dir: str,
    images_dir: str,
    class_names: List[str],
    train_images_dir: Optional[str] = None,
    val_images_dir: Optional[str] = None,
    test_images_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Convert COCO Segmentation using existing train/val/test split from Roboflow.

    This is the preferred method when Roboflow/Kaggle already provides separate
    train/val/test folders with their own annotations.

    Args:
        coco_json_path: Path to COCO format JSON file.
        output_dir: Output directory for YOLO segmentation dataset.
        images_dir: Base directory containing images.
        class_names: List of class names.
        train_images_dir: Path to training images (relative to images_dir or absolute).
        val_images_dir: Path to validation images.
        test_images_dir: Path to test images.

    Returns:
        Dict with paths info.
    """
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_map = {img["id"]: img for img in coco["images"]}
    categories_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations_by_image: Dict[int, List] = {img_id: [] for img_id in images_map}
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    category_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_id, img_info in tqdm(images_map.items(), desc="Converting annotations"):
        img_file_name = img_info["file_name"]
        img_path = Path(images_dir) / img_file_name

        split = "train"
        if "val" in img_file_name.lower() or "/val/" in img_file_name or "\\val\\" in img_file_name:
            split = "val"
        elif "test" in img_file_name.lower() or "/test/" in img_file_name or "\\test\\" in img_file_name:
            split = "test"

        img_width = img_info["width"]
        img_height = img_info["height"]

        label_file = output_dir / "labels" / split / (Path(img_file_name).stem + ".txt")
        with open(label_file, "w", encoding="utf-8") as f:
            for ann in annotations_by_image[img_id]:
                cat_name = categories_map[ann["category_id"]]
                if cat_name not in category_to_idx:
                    continue
                cls_idx = category_to_idx[cat_name]

                if "segmentation" in ann and ann["segmentation"]:
                    segmentation = ann["segmentation"][0]
                    normalized_coords = []

                    for i in range(0, len(segmentation), 2):
                        x = segmentation[i] / img_width
                        y = segmentation[i + 1] / img_height
                        normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])

                    coords_str = " ".join(normalized_coords)
                    f.write(f"{cls_idx} {coords_str}\n")

        dest_img = output_dir / "images" / split / img_file_name
        if img_path.exists():
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            img_data = cv2.imread(str(img_path))
            if img_data is not None:
                cv2.imwrite(str(dest_img), img_data)

    yaml_path = write_dataset_yaml(output_dir, class_names)
    return {"dataset_yaml": str(yaml_path)}


def write_dataset_yaml(output_dir: Path, class_names: List[str]) -> Path:
    """Write YOLO dataset.yaml file for segmentation.

    Args:
        output_dir: Dataset output directory.
        class_names: List of class names.

    Returns:
        Path to the created dataset.yaml.
    """
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# Durian Disease Segmentation - YOLO Dataset
# Auto-generated by converter_seg.py

path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nDataset YAML saved to: {yaml_path}")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO Segmentation annotations to YOLO Segmentation format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to COCO format JSON file with segmentation annotations",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/seg_yolo",
        help="Output directory for YOLO dataset (default: data/seg_yolo)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="Don't copy images to output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    class_names = [
        "ALGAL_LEAF_SPOT",
        "ALLOCARIDARA_ATTACK",
        "HEALTHY_LEAF",
        "LEAF_BLIGHT",
        "PHOMOPSIS_LEAF_SPOT",
    ]

    print("=" * 60)
    print("COCO SEGMENTATION → YOLO SEGMENTATION CONVERTER")
    print("=" * 60)
    print(f"  Input JSON: {args.input}")
    print(f"  Images dir: {args.images}")
    print(f"  Output dir: {args.output}")
    print(f"  Train/Val/Test: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print("=" * 60)

    result = coco_seg_to_yolo_seg(
        coco_json_path=args.input,
        output_dir=args.output,
        images_dir=args.images,
        class_names=class_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_images=not args.no_copy,
    )

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Dataset YAML: {result['dataset_yaml']}")
    print(f"  Training images: {result['num_train']}")
    print(f"  Validation images: {result['num_val']}")
    print(f"  Test images: {result['num_test']}")
    print("\nTo train YOLOv8 Segmentation:")
    print(f"  python scripts/train_segmentation.py --data {result['dataset_yaml']}")


if __name__ == "__main__":
    main()
