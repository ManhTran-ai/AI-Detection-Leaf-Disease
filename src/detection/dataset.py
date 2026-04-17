"""YOLO-format dataset for durian disease detection."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class YOLODataset(Dataset):
    """Dataset for YOLO-format object detection.

    Expects:
        root_dir/
            images/
                train/, val/, test/
            labels/
                train/, val/, test/  (same filenames as images, .txt)
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        class_names: Sequence[str],
        img_size: int = 640,
        transform=None,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.img_size = img_size
        self.transform = transform
        self.augment = augment

        self.img_dir = self.root_dir / "images" / split
        self.label_dir = self.root_dir / "labels" / split

        if not self.img_dir.exists():
            raise RuntimeError(f"Image directory not found: {self.img_dir}")

        self.image_files = sorted([f for f in self.img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])

        if not self.image_files:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Dict:
        img_path = self.image_files[index]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    boxes.append([xc, yc, bw, bh])
                    labels.append(cls_id)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels.tolist())
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels = np.array(transformed["class_labels"], dtype=np.int64) if transformed["class_labels"] else np.zeros((0,), dtype=np.int64)

        return {
            "image": img,
            "boxes": boxes,
            "labels": labels,
            "image_path": str(img_path),
            "orig_size": (orig_w, orig_h),
        }
