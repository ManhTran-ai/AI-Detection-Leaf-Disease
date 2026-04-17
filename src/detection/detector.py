"""Object detection predictor using YOLOv8."""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .utils import CLASS_NAMES, CLASS_COLORS, draw_detections, DISEASE_INFO_VN


class DetectionPredictor:
    """YOLOv8-based object detection predictor for durian disease detection."""

    def __init__(
        self,
        model_path: str,
        class_names: List[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            if torch.cuda.is_available():
                self.device = "0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def predict(
        self,
        image_path: str,
        conf_threshold: float = None,
        iou_threshold: float = None,
        return_image: bool = True,
    ) -> Dict:
        """Run object detection on a single image.

        Args:
            image_path: Path to input image.
            conf_threshold: Confidence threshold (overrides instance default).
            iou_threshold: IoU threshold for NMS (overrides instance default).
            return_image: If True, return annotated image.

        Returns:
            Dict with predictions, boxes, classes, confidences, and annotated image.
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        results = self.model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        predictions = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()

                predictions.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else "Unknown",
                    "confidence": conf,
                    "bbox": xyxy,
                })

        annotated_img = None
        if return_image:
            img = cv2.imread(image_path)
            if img is not None:
                boxes_list = [p["bbox"] for p in predictions]
                cls_ids = [p["class_id"] for p in predictions]
                confs = [p["confidence"] for p in predictions]
                annotated_img = draw_detections(img, boxes_list, cls_ids, confs, self.class_names, CLASS_COLORS)

        return {
            "predictions": predictions,
            "num_detections": len(predictions),
            "annotated_image": annotated_img,
            "image_path": image_path,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run object detection on a batch of images.

        Args:
            image_paths: List of image paths.

        Returns:
            List of prediction dicts.
        """
        results = self.model(image_paths, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        outputs = []
        for result in results:
            boxes = result.boxes
            predictions = []
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    predictions.append({
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else "Unknown",
                        "confidence": conf,
                        "bbox": xyxy,
                    })

            outputs.append({
                "predictions": predictions,
                "num_detections": len(predictions),
            })

        return outputs

    def get_disease_summary(self, predictions: List[Dict]) -> Dict:
        """Create a disease summary from predictions.

        Args:
            predictions: List of prediction dicts from predict().

        Returns:
            Summary dict with disease counts, most common disease, etc.
        """
        if not predictions or predictions.get("num_detections", 0) == 0:
            return {
                "has_disease": False,
                "diseases_found": [],
                "disease_counts": {},
                "primary_disease": None,
                "primary_confidence": 0.0,
            }

        disease_counts = {}
        for pred in predictions.get("predictions", []):
            name = pred["class_name"]
            disease_counts[name] = disease_counts.get(name, 0) + 1

        diseases_found = list(disease_counts.keys())
        has_disease = "HEALTHY_LEAF" not in diseases_found or len(diseases_found) > 1

        if disease_counts:
            primary = max(disease_counts.items(), key=lambda x: x[1])
            primary_info = DISEASE_INFO_VN.get(primary[0], {})
        else:
            primary = (None, 0)
            primary_info = {}

        return {
            "has_disease": has_disease,
            "diseases_found": diseases_found,
            "disease_counts": disease_counts,
            "primary_disease": primary[0],
            "primary_confidence": float(primary[1]) if primary[1] else 0.0,
            "primary_disease_info": primary_info,
        }
