import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import create_dataloaders
from src.evaluation.evaluator import Evaluator
from src.models.model_builder import build_model_components
from src.training.scheduler import create_scheduler
from src.training.trainer import Trainer
from src.utils.config import get_device, load_config, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Durian leaf disease detection model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--device", default=None, help="Force device, e.g. cpu or cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config).raw
    set_global_seed(config["experiment"].get("seed", 42))
    device = torch.device(args.device) if args.device else get_device()

    train_loader, val_loader, test_loader = create_dataloaders(config)
    model, optimizer, criterion = build_model_components(config, device)
    scheduler = create_scheduler(optimizer, config.get("scheduler", {}))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        scheduler=scheduler,
    )
    history = trainer.fit()

    evaluator = Evaluator(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=config["dataset"]["class_names"],
        save_dir=config["logging"]["metrics_dir"],
    )
    eval_results = evaluator.evaluate()
    print("Test metrics:", {k: v for k, v in eval_results.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()

