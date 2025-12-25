import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.visualization.plotting import plot_confusion_matrix, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Create plots from training and evaluation artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--history", default=None, help="Optional explicit history JSON path.")
    parser.add_argument("--evaluation", default=None, help="Optional explicit evaluation results path.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = load_config(args.config).raw
        logging_cfg = config["logging"]

        history_path = Path(args.history) if args.history else Path(logging_cfg["metrics_dir"]) / "training_history.json"
        evaluation_path = (
            Path(args.evaluation) if args.evaluation else Path(logging_cfg["metrics_dir"]) / "evaluation_results.pt"
        )
        plots_dir = Path(logging_cfg["plots_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading training history from: {history_path}")
        if not history_path.exists():
            print(f"❌ Error: File not found: {history_path}")
            return
        
        with open(history_path, "r", encoding="utf-8") as fp:
            history = json.load(fp)
        
        print("Creating training curves...")
        plot_training_curves(history, str(plots_dir))
        training_curve_path = plots_dir / "training_curves.png"
        if training_curve_path.exists():
            print(f"✓ Training curves saved to: {training_curve_path}")
        else:
            print(f"❌ Failed to save training curves to: {training_curve_path}")

        print(f"Loading evaluation results from: {evaluation_path}")
        if not evaluation_path.exists():
            print(f"❌ Error: File not found: {evaluation_path}")
            return
        
        evaluation = torch.load(evaluation_path, map_location='cpu')
        print("Creating confusion matrix...")
        plot_confusion_matrix(evaluation["confusion_matrix"], config["dataset"]["class_names"], str(plots_dir))
        confusion_matrix_path = plots_dir / "confusion_matrix.png"
        if confusion_matrix_path.exists():
            print(f"✓ Confusion matrix saved to: {confusion_matrix_path}")
        else:
            print(f"❌ Failed to save confusion matrix to: {confusion_matrix_path}")
        
        print(f"\n✅ All plots saved under {plots_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

