"""Utilities for object detection."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

CLASS_NAMES = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT",
]

CLASS_COLORS = {
    "ALGAL_LEAF_SPOT": (0, 200, 0),
    "ALLOCARIDARA_ATTACK": (255, 165, 0),
    "HEALTHY_LEAF": (0, 255, 0),
    "LEAF_BLIGHT": (255, 0, 0),
    "PHOMOPSIS_LEAF_SPOT": (200, 0, 200),
}

SEGMENTATION_COLORS = {
    0: (0, 200, 0),       # ALGAL_LEAF_SPOT - Green
    1: (255, 165, 0),     # ALLOCARIDARA_ATTACK - Orange
    2: (0, 255, 0),       # HEALTHY_LEAF - Bright Green
    3: (255, 0, 0),       # LEAF_BLIGHT - Red
    4: (200, 0, 200),     # PHOMOPSIS_LEAF_SPOT - Purple
}

DISEASE_INFO_VN = {
    "ALGAL_LEAF_SPOT": {
        "name": "Bệnh đốm tảo",
        "description": "Bệnh đốm tảo (Cephaleuros virescens) tạo ra các đốm tròn màu xanh xám hoặc nâu đỏ trên bề mặt lá.",
        "treatment": "Cắt tỉa lá bệnh, cải thiện thông gió, phun thuốc gốc đồng.",
    },
    "ALLOCARIDARA_ATTACK": {
        "name": "Bọ trĩ tấn công",
        "description": "Bọ trĩ (Allocaridara malayensis) hút nhựa từ lá non, làm lá bị quăn và biến dạng.",
        "treatment": "Phun thuốc trừ sâu như Imidacloprid, Abamectin.",
    },
    "HEALTHY_LEAF": {
        "name": "Lá khỏe mạnh",
        "description": "Lá có màu xanh đậm đồng đều, bóng mượt, không có dấu hiệu bệnh.",
        "treatment": None,
    },
    "LEAF_BLIGHT": {
        "name": "Bệnh cháy lá",
        "description": "Bệnh cháy lá làm lá chuyển màu nâu từ mép, sau đó lan rộng và khô héo.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc diệt nấm Mancozeb, Metalaxyl.",
    },
    "PHOMOPSIS_LEAF_SPOT": {
        "name": "Bệnh đốm lá Phomopsis",
        "description": "Nấm Phomopsis gây ra các đốm tròn màu nâu với viền đậm hơn.",
        "treatment": "Phun thuốc diệt nấm như Carbendazim, Thiophanate-methyl.",
    },
}


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    class_ids: List[int],
    confidences: List[float],
    class_names: List[str] = None,
    colors: dict = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on image.

    Args:
        image: Image array (H, W, C) in BGR format.
        boxes: List of boxes in [x1, y1, x2, y2] format (pixels).
        class_ids: List of class IDs.
        confidences: List of confidence scores.
        class_names: List of class names.
        colors: Dict mapping class_id or class_name to BGR color.
        thickness: Line thickness.
        font_scale: Font scale for text.

    Returns:
        Image with drawn boxes.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if colors is None:
        colors = CLASS_COLORS

    result = image.copy()
    h, w = result.shape[:2]

    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if isinstance(cls_id, str):
            color = colors.get(cls_id, (255, 255, 255))
        else:
            name = class_names[cls_id] if cls_id < len(class_names) else "Unknown"
            color = colors.get(name, (255, 255, 255))

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_names[cls_id]}: {conf:.2f}" if cls_id < len(class_names) else f"Class {cls_id}: {conf:.2f}"

        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y1 = max(y1 - label_h - baseline - 4, 0)
        cv2.rectangle(result, (x1, label_y1), (x1 + label_w, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return result


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO format label file.

    Args:
        label_path: Path to .txt label file.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of (class_id, x_center, y_center, width, height) tuples, all in pixels.
    """
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc = float(parts[1]) * img_width
            yc = float(parts[2]) * img_height
            bw = float(parts[3]) * img_width
            bh = float(parts[4]) * img_height

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
            annotations.append((cls_id, x1, y1, x2, y2))
    return annotations


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Array of shape (N, 4) in xywh format.

    Returns:
        Array of shape (N, 4) in xyxy format.
    """
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        IoU score.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
