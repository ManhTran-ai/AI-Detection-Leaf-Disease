"""Segmentation predictor for instance segmentation with YOLOv8."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from .utils import CLASS_NAMES, SEGMENTATION_COLORS


class SegmentationPredictor:
    """Lazy-loading segmentation predictor for YOLOv8 Instance Segmentation."""

    _cache: Dict[str, "SegmentationPredictor"] = {}

    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initialize segmentation predictor.

        Args:
            model_path: Path to YOLO segmentation model (.pt).
            class_names: List of class names.
            conf_threshold: Confidence threshold for predictions.
            iou_threshold: IoU threshold for NMS.
            device: Device to use ('0', '1', 'cpu').
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))

    @classmethod
    def get_predictor(
        cls,
        model_path: str,
        class_names: List[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> "SegmentationPredictor":
        """Get or create a cached predictor instance."""
        cache_key = f"{model_path}_{conf_threshold}_{iou_threshold}_{device}"

        if cache_key not in cls._cache:
            cls._cache[cache_key] = cls(
                model_path=model_path,
                class_names=class_names,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )

        return cls._cache[cache_key]

    def predict(
        self,
        image_path: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        return_image: bool = True,
    ) -> Dict:
        """Run segmentation prediction on an image.

        Args:
            image_path: Path to image or numpy array.
            conf_threshold: Override confidence threshold.
            iou_threshold: Override IoU threshold.
            return_image: Whether to return annotated image.

        Returns:
            Dict with predictions, masks, and optionally annotated image.
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        results = self._model(
            image_path,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        predictions = []
        polygons = []
        masks_data = []

        if masks is not None:
            mask_xy = masks.xy
            mask_classes = masks.cls
            mask_conf = masks.conf

            for i in range(len(mask_xy)):
                cls_id = int(mask_classes[i])
                conf_score = float(mask_conf[i])
                polygon = mask_xy[i]

                predictions.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else "Unknown",
                    "confidence": conf_score,
                    "polygon": polygon.tolist(),
                    "bbox": boxes.xyxy[i].tolist() if boxes is not None else None,
                })

                polygons.append(polygon)
                masks_data.append({
                    "data": masks.data[i].cpu().numpy() if hasattr(masks.data, "cpu") else masks.data[i],
                    "confidence": conf_score,
                    "class_id": cls_id,
                })

        annotated_image = None
        if return_image:
            if isinstance(image_path, np.ndarray):
                img = image_path.copy()
            else:
                img = cv2.imread(str(image_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                annotated_image = self._draw_masks(img, polygons, predictions)

        return {
            "predictions": predictions,
            "num_detections": len(predictions),
            "polygons": polygons,
            "masks": masks_data,
            "annotated_image": annotated_image,
        }

    def _draw_masks(
        self,
        image: np.ndarray,
        polygons: List[np.ndarray],
        predictions: List[Dict],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw segmentation masks on image.

        Args:
            image: Image in RGB format.
            polygons: List of polygon arrays.
            predictions: List of prediction dicts.
            alpha: Mask transparency.

        Returns:
            Annotated image in RGB format.
        """
        img = image.copy()

        for i, (polygon, pred) in enumerate(zip(polygons, predictions)):
            cls_id = pred["class_id"]
            color = SEGMENTATION_COLORS.get(cls_id, (255, 255, 255))

            pts = polygon.astype(np.int32).reshape((-1, 1, 2))

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (*color, int(255 * alpha)))
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            cv2.polylines(img, [pts], True, color, 2)

        return img

    @classmethod
    def clear_cache(cls):
        """Clear the predictor cache."""
        cls._cache.clear()


def draw_segmentation_masks(
    image: np.ndarray,
    predictions: List[Dict],
    class_names: List[str] = None,
    colors: dict = None,
    alpha: float = 0.4,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw segmentation masks with labels on image.

    Args:
        image: Image in RGB format (H, W, 3).
        predictions: List of prediction dicts with 'polygon', 'class_id', 'confidence'.
        class_names: List of class names.
        colors: Dict mapping class_id to BGR color.
        alpha: Mask transparency (0-1).
        thickness: Line thickness for polygon outlines.
        font_scale: Font scale for labels.
        show_labels: Whether to show class labels.

    Returns:
        Annotated image in RGB format.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if colors is None:
        colors = SEGMENTATION_COLORS

    img = image.copy()

    for pred in predictions:
        cls_id = pred["class_id"]
        conf = pred.get("confidence", 1.0)
        polygon = pred["polygon"]

        if isinstance(polygon, list):
            polygon = np.array(polygon)

        color = colors.get(cls_id, (255, 255, 255))

        pts = polygon.astype(np.int32).reshape((-1, 1, 2))

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (*color, int(255 * alpha)))
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.polylines(img, [pts], True, color, thickness)

        if show_labels:
            center_x = int(np.mean(polygon[:, 0]))
            center_y = int(np.mean(polygon[:, 1]))
            label = f"{class_names[cls_id]}: {conf:.2f}"

            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
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
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return img
