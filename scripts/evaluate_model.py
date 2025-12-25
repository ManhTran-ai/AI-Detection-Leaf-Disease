import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import create_dataloaders
from src.evaluation.evaluator import Evaluator
from src.models.model_factory import build_model
from src.models.utils import load_checkpoint
from src.utils.config import get_device, load_config, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config).raw
    set_global_seed(config["experiment"].get("seed", 42))
    device = torch.device(args.device) if args.device else get_device()

    train_loader, val_loader, test_loader = create_dataloaders(config)
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataloader = split_map[args.split]

    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=config["dataset"]["class_names"],
        save_dir=config["logging"]["metrics_dir"],
    )
    metrics = evaluator.evaluate()
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

