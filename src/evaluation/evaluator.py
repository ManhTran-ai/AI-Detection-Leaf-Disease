from pathlib import Path
from typing import Dict, List, Optional

import torch

from .metrics import compute_metrics


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        class_names: List[str],
        save_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> Dict:
        self.model.eval()
        preds: List[int] = []
        labels: List[int] = []
        probabilities: List[List[float]] = []

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                outputs = self.model(images)
                probabilities.extend(torch.softmax(outputs, dim=1).cpu().tolist())
                preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())

        results = compute_metrics(labels, preds, self.class_names)
        results["probabilities"] = probabilities
        results["predictions"] = preds
        results["labels"] = labels
        if self.save_dir:
            torch.save(results, self.save_dir / "evaluation_results.pt")
        return results


