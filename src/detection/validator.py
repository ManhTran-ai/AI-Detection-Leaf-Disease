"""Detection validator for mAP computation and IoU-based evaluation."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class DetectionValidator:
    """Validate object detection predictions against ground truth boxes."""

    def __init__(
        self,
        class_names: List[str],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def validate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict:
        """Validate predictions against ground truth.

        Args:
            predictions: List of dicts with 'boxes' ([N,4]), 'labels' ([N,]) per image.
            ground_truths: List of dicts with 'boxes', 'labels' per image.

        Returns:
            Metrics dict with TP, FP, FN, precision, recall, mAP.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_ious = []

        per_class_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(self.num_classes)}

        for pred_dict, gt_dict in zip(predictions, ground_truths):
            pred_boxes = np.array(pred_dict.get("boxes", []))
            pred_labels = np.array(pred_dict.get("labels", []))
            gt_boxes = np.array(gt_dict.get("boxes", []))
            gt_labels = np.array(gt_dict.get("labels", []))

            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                for lbl in pred_labels:
                    if lbl < self.num_classes:
                        per_class_stats[int(lbl)]["fp"] += 1
                continue

            if len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                for lbl in gt_labels:
                    if lbl < self.num_classes:
                        per_class_stats[int(lbl)]["fn"] += 1
                continue

            matched_gt = set()
            for i, (pbox, plbl) in enumerate(zip(pred_boxes, pred_labels)):
                best_iou = 0.0
                best_gt_idx = -1
                for j, (gbox, glbl) in enumerate(zip(gt_boxes, gt_labels)):
                    if int(plbl) != int(glbl) or j in matched_gt:
                        continue
                    iou = self._compute_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    total_fp += 0
                    matched_gt.add(best_gt_idx)
                    all_ious.append(best_iou)
                    lbl = int(plbl) if plbl < self.num_classes else 0
                    per_class_stats[lbl]["tp"] += 1
                else:
                    total_fp += 1
                    lbl = int(plbl) if plbl < self.num_classes else 0
                    per_class_stats[lbl]["fp"] += 1

            unmatched_gt = set(range(len(gt_boxes))) - matched_gt
            total_fn += len(unmatched_gt)
            for j in unmatched_gt:
                lbl = int(gt_labels[j]) if gt_labels[j] < self.num_classes else 0
                per_class_stats[lbl]["fn"] += 1

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

        per_class_metrics = {}
        for cls_id, stats in per_class_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class_{cls_id}"
            per_class_metrics[cls_name] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "per_class": per_class_metrics,
        }

    def compute_ap(self, predictions: List[Dict], ground_truths: List[Dict], cls_id: int) -> float:
        """Compute Average Precision for a single class at IoU threshold.

        Args:
            predictions: List of prediction dicts per image.
            ground_truths: List of GT dicts per image.
            cls_id: Class ID to compute AP for.

        Returns:
            AP@IoU score.
        """
        all_detections = []
        total_gt = 0

        for img_idx, (pred_dict, gt_dict) in enumerate(zip(predictions, ground_truths)):
            pred_boxes = np.array(pred_dict.get("boxes", []))
            pred_labels = np.array(pred_dict.get("labels", []))
            pred_confs = np.array(pred_dict.get("confidences", []))

            gt_boxes = np.array(gt_dict.get("boxes", []))
            gt_labels = np.array(gt_dict.get("labels", []))

            cls_gt_mask = gt_labels == cls_id
            cls_gt_boxes = gt_boxes[cls_gt_mask]
            total_gt += len(cls_gt_boxes)
            matched = set()

            cls_pred_mask = pred_labels == cls_id
            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_confs = pred_confs[cls_pred_mask] if len(pred_confs) > 0 else np.ones(len(cls_pred_boxes))

            if len(cls_pred_boxes) > 0:
                sorted_indices = np.argsort(-cls_pred_confs)
                for pred_idx in sorted_indices:
                    pbox = cls_pred_boxes[pred_idx]
                    best_iou = 0.0
                    best_gt_idx = -1
                    for gt_idx, gbox in enumerate(cls_gt_boxes):
                        if gt_idx in matched:
                            continue
                        iou = self._compute_iou(pbox, gbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    tp = 1 if (best_iou >= self.iou_threshold and best_gt_idx >= 0) else 0
                    if tp:
                        matched.add(best_gt_idx)
                    all_detections.append({"img_idx": img_idx, "conf": cls_pred_confs[pred_idx], "tp": tp})

        if total_gt == 0:
            return 0.0

        if not all_detections:
            return 0.0

        sorted_dets = sorted(all_detections, key=lambda x: -x["conf"])
        cumsum_tp = np.cumsum([d["tp"] for d in sorted_dets])
        cumsum_fp = np.cumsum([1 - d["tp"] for d in sorted_dets])

        recalls = cumsum_tp / total_gt
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)

        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]

        return float(ap)
