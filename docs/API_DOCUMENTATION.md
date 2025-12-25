# Durian Leaf Disease Detection - API Documentation

## Data Module (`src/data`)

- `DurianLeafDataset(root_dir, class_names, transforms=None, return_paths=False)`
  - Loads images from `<root_dir>/<class_name>` folders.
  - Returns dict with `image`, `label`, optionally `path`.
- `build_train_transforms(config)` / `build_eval_transforms(config)`
  - Albumentations pipelines driven by YAML augmentation settings.
- `create_dataloaders(config)`
  - Returns `(train_loader, val_loader, test_loader)` with weighted sampler for class balancing.

## Models (`src/models`)

- `create_resnet(name, num_classes, dropout, pretrained)`
  - Wraps torchvision ResNet18/34/50 with configurable head.
- `CustomCNN`
  - Configurable 4-layer ConvNet with BatchNorm, MaxPooling, Dropout.
- `build_model(config)`
  - Factory returning the correct architecture based on `model.name`.
- `build_model_components(config, device)`
  - Returns `(model, optimizer, criterion)` using training hyperparameters.
- `save_checkpoint`, `load_checkpoint`, `count_parameters`
  - Utility helpers for persistence and reporting.

## Training (`src/training`)

- `Trainer`
  - Handles training loop, validation, TensorBoard logging, AMP, early stopping, and checkpointing.
- `accuracy`, `compute_epoch_metrics`
- `EarlyStopping`, `CheckpointManager`
- `create_scheduler`

## Evaluation (`src/evaluation`)

- `compute_metrics(labels, preds, class_names)`
  - Generates accuracy, macro precision/recall/F1, confusion matrix, and classification report.
- `Evaluator`
  - Runs inference across a dataloader and persists evaluation artifacts.

## Inference (`src/inference`)

- `preprocess_image(image_path, dataset_cfg)`
  - Loads a single image, applies resize/normalization, returns tensor and original array.
- `Predictor`
  - Loads model checkpoint and produces class probabilities for an image.

## Visualization (`src/visualization`)

- `plot_training_curves(history, output_dir)` / `plot_confusion_matrix(cm, class_names, output_dir)`
- `generate_grad_cam(model, target_layer, input_tensor, original_image, class_idx, output_path)`
  - Produces Grad-CAM overlay for interpretability.

## Scripts

- `scripts/train_model.py` – end-to-end training + evaluation entry point.
- `scripts/evaluate_model.py` – evaluate a checkpoint on train/val/test split.
- `scripts/inference.py` – single image prediction with optional Grad-CAM.
- `scripts/split_dataset.py` – stratified directory split helper.
- `scripts/visualize_results.py` – generate plots after training.


