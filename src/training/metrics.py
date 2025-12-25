from typing import Dict, List, Sequence

import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).float().sum()
    return (correct / targets.numel()).item()


def compute_epoch_metrics(preds: Sequence[int], labels: Sequence[int], average: str = "macro") -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average, zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


