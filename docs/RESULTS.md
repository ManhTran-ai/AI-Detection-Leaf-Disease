# Durian Leaf Disease Detection - Results Summary

| Model       | Accuracy (Target) | Notes |
|-------------|-------------------|-------|
| ResNet18    | 82-88%            | Baseline transfer learning configuration. |
| ResNet34    | 88-92%            | Stronger backbone with extended training schedule. |
| ResNet50    | >92%              | High-capacity model with mixed precision support. |
| Custom CNN  | Baseline          | Lightweight 4-layer CNN for comparison. |

## Logging Artifacts

- Training history JSON saved to `results/<model>/metrics/training_history.json`
- TensorBoard logs under `results/<model>/tensorboard`
- Evaluation summaries saved to `results/<model>/metrics/evaluation_results.pt`
- Plots (loss, accuracy, confusion matrix) generated via `scripts/visualize_results.py` and stored in `results/<model>/plots`

## How to Update

1. Train a model using `scripts/train_model.py --config configs/config_resnet18.yaml`
2. After training, run `scripts/visualize_results.py --config configs/config_resnet18.yaml`
3. Update this file with the actual numbers obtained from `evaluation_results.pt`


