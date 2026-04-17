"""Detection metrics: mAP, precision, recall, F1 for object detection."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


class DetectionMetrics:
    """Compute detection metrics (mAP, precision, recall) using Ultralytics."""

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate(
        self,
        model_path: str,
        data_yaml: str,
        split: str = "test",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.6,
    ) -> Dict:
        """Evaluate detection model and return metrics.

        Args:
            model_path: Path to trained YOLO model (.pt).
            data_yaml: Path to dataset.yaml.
            split: Dataset split to evaluate on.
            conf_threshold: Confidence threshold for evaluation.
            iou_threshold: IoU threshold for evaluation.

        Returns:
            Dict with mAP, precision, recall, F1 per class and overall.
        """
        model = YOLO(model_path)
        metrics = model.val(
            data=data_yaml,
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True,
        )

        per_class_metrics = {}
        if hasattr(metrics, "ap_class_index") and metrics.ap_class_index is not None:
            for idx, cls_idx in enumerate(metrics.ap_class_index):
                if cls_idx < len(self.class_names):
                    cls_name = self.class_names[cls_idx]
                    per_class_metrics[cls_name] = {
                        "AP50": float(metrics.box.ap50[cls_idx]) if hasattr(metrics.box, "ap50") else 0.0,
                        "AP": float(metrics.box.ap[cls_idx]) if hasattr(metrics.box, "ap") else 0.0,
                        "Precision": float(metrics.box.p[cls_idx]) if hasattr(metrics.box, "p") else 0.0,
                        "Recall": float(metrics.box.r[cls_idx]) if hasattr(metrics.box, "r") else 0.0,
                    }

        result = {
            "map50": float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0,
            "map50_95": float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0,
            "precision": float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0,
            "recall": float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0,
            "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)) if hasattr(metrics.box, "mp") else 0.0,
            "per_class": per_class_metrics,
            "num_gt": int(metrics.box.nt.sum()) if hasattr(metrics.box, "nt") else 0,
        }

        return result

    def print_metrics(self, metrics: Dict) -> None:
        """Pretty-print detection metrics."""
        print("\n" + "=" * 60)
        print("DETECTION EVALUATION RESULTS")
        print("=" * 60)
        print(f"  mAP@0.50     : {metrics['map50']:.4f}")
        print(f"  mAP@0.50:0.95: {metrics['map50_95']:.4f}")
        print(f"  Precision    : {metrics['precision']:.4f}")
        print(f"  Recall       : {metrics['recall']:.4f}")
        print(f"  F1 Score     : {metrics['f1']:.4f}")
        print(f"  GT Boxes     : {metrics['num_gt']}")
        print("-" * 60)
        print("Per-Class Metrics:")
        print(f"  {'Class':<25} {'AP50':>8} {'AP':>8} {'P':>8} {'R':>8} {'F1':>8}")
        print("  " + "-" * 65)
        for cls_name, m in metrics.get("per_class", {}).items():
            p = m.get("Precision", 0)
            r = m.get("Recall", 0)
            f1 = 2 * p * r / (p + r + 1e-8)
            print(f"  {cls_name:<25} {m.get('AP50', 0):>8.4f} {m.get('AP', 0):>8.4f} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")
        print("=" * 60 + "\n")
