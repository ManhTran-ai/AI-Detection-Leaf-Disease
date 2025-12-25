from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class DurianLeafDataset(Dataset):
    """Dataset wrapper around a simple class-folder image structure."""

    def __init__(
        self,
        root_dir: str,
        class_names: Sequence[str],
        transforms=None,
        return_paths: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.transforms = transforms
        self.return_paths = return_paths
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}. Please verify the dataset paths.")

    def _gather_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for cls in self.class_names:
            class_dir = self.root_dir / cls
            if not class_dir.exists():
                continue
            for img_path in class_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((str(img_path), self.class_to_idx[cls]))
        samples.sort(key=lambda item: item[0])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image_np)
            image_tensor = transformed["image"]
        else:
            image_tensor = image_np

        sample = {"image": image_tensor, "label": label}
        if self.return_paths:
            sample["path"] = image_path
        return sample


