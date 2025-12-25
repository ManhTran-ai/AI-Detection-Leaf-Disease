import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import IMG_EXTENSIONS  # noqa: E402
from src.data.preprocessing import build_train_transforms  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _gather_images(root_dir: Path, class_names, limit: int):
    images = []
    for cls in class_names:
        cls_dir = root_dir / cls
        if not cls_dir.exists():
            continue
        images.extend(
            [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        )
    random.shuffle(images)
    return images[:limit]


def _denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    image = tensor.clone()
    for channel, (m, s) in enumerate(zip(mean, std)):
        image[channel] = image[channel] * s + m
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training augmentations.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--output",
        default="outputs/demo_results/augmentations.png",
        help="Path to save the visualization grid.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    config = load_config(args.config).raw
    dataset_cfg = config["dataset"]
    transform = build_train_transforms(config)

    train_dir = Path(dataset_cfg["train_dir"])
    class_names = dataset_cfg["class_names"]
    samples = _gather_images(train_dir, class_names, args.num_samples)
    if not samples:
        raise RuntimeError(f"No images found under {train_dir}. Cannot visualize augmentations.")

    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))

    for idx, img_path in enumerate(samples):
        image = np.array(Image.open(img_path).convert("RGB"))
        aug_image = transform(image=image)["image"]

        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f"Original ({Path(img_path).parent.name})")
        axes[idx, 0].axis("off")

        if isinstance(aug_image, torch.Tensor):
            aug_image = _denormalize(aug_image, dataset_cfg["mean"], dataset_cfg["std"])

        axes[idx, 1].imshow(aug_image)
        axes[idx, 1].set_title("Augmented")
        axes[idx, 1].axis("off")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved augmentation preview to {output_path}")


if __name__ == "__main__":
    main()


