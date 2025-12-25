from typing import Dict

import torch

from src.inference.preprocessing import preprocess_image
from src.models.model_factory import build_model
from src.models.utils import load_checkpoint


class Predictor:
    def __init__(self, config: Dict, checkpoint_path: str, device: torch.device) -> None:
        self.config = config
        self.dataset_cfg = config["dataset"]
        self.augmentation_cfg = config.get("augmentation", {})
        self.class_names = self.dataset_cfg["class_names"]
        self.device = device

        self.model = build_model(config).to(device)
        load_checkpoint(self.model, checkpoint_path, device)
        self.model.eval()

    def predict(self, image_path: str) -> Dict:
        tensor, original_image = preprocess_image(
            image_path, self.dataset_cfg, augmentation_cfg=self.augmentation_cfg
        )
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        confidence = float(probabilities.max())
        predicted_idx = int(probabilities.argmax())
        return {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": confidence,
            "probabilities": {
                cls_name: float(probabilities[idx]) for idx, cls_name in enumerate(self.class_names)
            },
            "image": original_image,
            "input_tensor": tensor.squeeze(0).cpu(),
        }

