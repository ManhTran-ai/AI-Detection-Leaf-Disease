from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_confusion_matrix(cm, class_names: List[str], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


