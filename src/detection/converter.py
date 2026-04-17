"""Convert COCO/VOC annotations to YOLO TXT format."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    images_dir: str,
    class_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    copy_images: bool = True,
) -> Dict[str, str]:
    """Convert COCO JSON annotations to YOLO TXT format.

    Args:
        coco_json_path: Path to COCO format JSON file.
        output_dir: Output directory for YOLO dataset.
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

    for img_id, img_info in tqdm(images_map.items(), desc="Converting annotations"):
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

                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                    xc = (x + w / 2) / img_width
                    yc = (y + h / 2) / img_height
                    nw = w / img_width
                    nh = h / img_height
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        if copy_images and img_path.exists():
            dest_img = output_dir / "images" / split / img_file_name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest_img), cv2.imread(str(img_path)))

    yaml_path = write_dataset_yaml(output_dir, class_names)

    return {
        "dataset_yaml": str(yaml_path),
        "train_dir": str(splits["train"]),
        "val_dir": str(splits["val"]),
        "test_dir": str(splits["test"]),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
    }


def voc_to_yolo(
    voc_annotations_dir: str,
    voc_images_dir: str,
    output_dir: str,
    class_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, str]:
    """Convert Pascal VOC XML annotations to YOLO TXT format.

    Args:
        voc_annotations_dir: Directory containing VOC XML annotation files.
        voc_images_dir: Directory containing images.
        output_dir: Output directory for YOLO dataset.
        class_names: List of class names.
        train_ratio: Ratio of training images.
        val_ratio: Ratio of validation images.
        test_ratio: Ratio of test images.

    Returns:
        Dict with paths to dataset.yaml and split directories.
    """
    import xml.etree.ElementTree as ET

    voc_annotations_dir = Path(voc_annotations_dir)
    voc_images_dir = Path(voc_images_dir)
    output_dir = Path(output_dir)

    category_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_files = []
    for xml_file in sorted(voc_annotations_dir.glob("*.xml")):
        img_file = voc_images_dir / (xml_file.stem + ".jpg")
        if not img_file.exists():
            img_file = voc_images_dir / (xml_file.stem + ".png")
        if not img_file.exists():
            img_file = voc_images_dir / (xml_file.stem + ".jpeg")
        if img_file.exists():
            all_files.append((xml_file, img_file))

    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    splits_files = {
        "train": [all_files[i] for i in indices[:n_train]],
        "val": [all_files[i] for i in indices[n_train : n_train + n_val]],
        "test": [all_files[i] for i in indices[n_train + n_val :]],
    }

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, file_pairs in splits_files.items():
        for xml_file, img_file in tqdm(file_pairs, desc=f"Converting {split}"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find("size")
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            label_file = output_dir / "labels" / split / (xml_file.stem + ".txt")
            with open(label_file, "w", encoding="utf-8") as f:
                for obj in root.findall("object"):
                    cls_name = obj.find("name").text
                    if cls_name not in category_to_idx:
                        continue
                    cls_idx = category_to_idx[cls_name]

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    xc = ((xmin + xmax) / 2) / img_width
                    yc = ((ymin + ymax) / 2) / img_height
                    nw = (xmax - xmin) / img_width
                    nh = (ymax - ymin) / img_height
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            dest_img = output_dir / "images" / split / img_file.name
            cv2.imwrite(str(dest_img), cv2.imread(str(img_file)))

    yaml_path = write_dataset_yaml(output_dir, class_names)
    return {
        "dataset_yaml": str(yaml_path),
        "num_train": len(splits_files["train"]),
        "num_val": len(splits_files["val"]),
        "num_test": len(splits_files["test"]),
    }


def write_dataset_yaml(output_dir: Path, class_names: List[str]) -> Path:
    """Write YOLO dataset.yaml file.

    Args:
        output_dir: Dataset output directory.
        class_names: List of class names.

    Returns:
        Path to the created dataset.yaml.
    """
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""# Durian Disease Detection - YOLO Dataset
# Auto-generated by converter.py

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

    return yaml_path


def create_dataset_from_class_folders(
    source_dir: str,
    output_dir: str,
    class_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
) -> Dict[str, str]:
    """Create YOLO dataset from class-folder structure (like existing classification dataset).

    Assumes source_dir contains subfolders named by class names, each with images.

    Args:
        source_dir: Source directory with class subfolders.
        output_dir: Output directory for YOLO dataset.
        class_names: List of class names.
        train_ratio: Ratio of training images.
        val_ratio: Ratio of validation images.
        test_ratio: Ratio of test images.
        image_extensions: Valid image extensions.

    Returns:
        Dict with paths info.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    all_images = []
    for cls_name in class_names:
        cls_dir = source_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_file in cls_dir.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                all_images.append((img_file, cls_name))

    if not all_images:
        raise RuntimeError(f"No images found in {source_dir}")

    cls_idx_map = {name: idx for idx, name in enumerate(class_names)}

    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    splits = {
        "train": [all_images[i] for i in indices[:n_train]],
        "val": [all_images[i] for i in indices[n_train : n_train + n_val]],
        "test": [all_images[i] for i in indices[n_train + n_val :]],
    }

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        for img_path, cls_name in tqdm(items, desc=f"Creating {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            img_stem = img_path.stem

            label_file = output_dir / "labels" / split / (img_stem + ".txt")
            with open(label_file, "w", encoding="utf-8") as f:
                cls_idx = cls_idx_map[cls_name]
                f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

            dest_img = output_dir / "images" / split / (img_stem + img_path.suffix)
            cv2.imwrite(str(dest_img), img)

    yaml_path = write_dataset_yaml(output_dir, class_names)
    return {
        "dataset_yaml": str(yaml_path),
        "num_train": len(splits["train"]),
        "num_val": len(splits["val"]),
        "num_test": len(splits["test"]),
    }
