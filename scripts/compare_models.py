r"""Compare CNN (Softmax) vs CNN+SVM classification for durian leaf disease detection.

This script orchestrates the full pipeline:
  1. Load dataset (train / val / test splits)
  2. Extract features from all 5 CNN models (with caching)
  3. Train + tune SVM for each CNN feature set
  4. Evaluate CNN-Softmax and CNN+SVM for each model
  5. Generate comparison charts and save results

Usage:
    python scripts/compare_models.py                           # All models
    python scripts/compare_models.py --models resnet50        # Single model
    python scripts/compare_models.py --no-svm                  # CNN-only evaluation
    python scripts/compare_models.py --no-cnn                  # SVM-only evaluation
    python scripts/compare_models.py --models resnet50 resnet18
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.svm.config import (
    DISEASE_CLASSES,
    FEATURE_CACHE_DIR,
    MODEL_REGISTRY,
    RANDOM_STATE,
    RESULTS_CM_DIR,
    RESULTS_DIR,
    SVM_MODELS_DIR,
    get_all_model_names,
    get_dataset_paths,
    get_model_info,
)
from src.svm.evaluate import (
    evaluate_model,
    plot_comparison_chart,
    plot_confusion_matrix,
    plot_roc_curves,
    print_results_table,
    save_results_csv,
)
from src.svm.feature_extractor import CNNFeatureExtractor, extract_all_splits
from src.svm.svm_pipeline import SVMClassifier
from src.utils.config import get_device, load_config, set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CNN softmax evaluation helpers
# ---------------------------------------------------------------------------

def _build_cnn_model(model_name: str, device: torch.device):
    """Build a fine-tuned CNN model and load its checkpoint."""
    model_info = get_model_info(model_name)
    config = load_config(model_info.config_path).raw
    from src.models.model_factory import build_model
    from src.models.utils import load_checkpoint

    model = build_model(config).to(device)
    if model_info.checkpoint_exists():
        load_checkpoint(model, model_info.checkpoint_path, device)
        logger.info(f"Loaded CNN checkpoint: {model_info.checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {model_info.checkpoint_path}. Using random weights.")
    model.eval()
    return model


def _evaluate_cnn_softmax(
    model_name: str,
    device: torch.device,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate a CNN model using softmax predictions."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    model = _build_cnn_model(model_name, device)
    dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    t0 = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    elapsed = (time.time() - t0) / len(y_test) * 1000  # ms per sample

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    acc = accuracy_score(y_test, all_preds)
    prec, rec, f1_macro, _ = precision_recall_fscore_support(y_test, all_preds, average="macro", zero_division=0)
    prec_w, rec_w, f1_weighted, _ = precision_recall_fscore_support(y_test, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, all_preds)

    try:
        probs_arr = np.array(all_probs)
        if probs_arr.shape[1] == 1:
            probs_arr = np.column_stack([1 - probs_arr[:, 0], probs_arr[:, 0]])
        auc = roc_auc_score(y_test, probs_arr, multi_class="ovr", average="macro", labels=list(range(len(class_names))))
    except Exception:
        auc = None

    results = {
        "model_name": get_model_info(model_name).display_name,
        "mode": "cnn",
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "roc_auc": float(auc) if auc is not None else None,
        "inference_ms": float(elapsed),
        "confusion_matrix": cm,
        "num_test_samples": len(y_test),
        "labels": y_test.tolist(),
        "predictions": all_preds,
        "probabilities": all_probs,
    }
    logger.info(
        f"[{model_name}-CNN] Acc={acc:.4f} | F1-macro={f1_macro:.4f} | "
        f"Precision={prec:.4f} | Recall={rec:.4f}" + (f" | ROC-AUC={auc:.4f}" if auc is not None else "")
    )
    return results


# ---------------------------------------------------------------------------
# SVM evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_svm_classifier(
    clf: SVMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate an SVM classifier on test data."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=list(range(len(class_names))))
    except Exception:
        auc = None

    results = {
        "model_name": model_name,
        "mode": "svm",
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "roc_auc": float(auc) if auc is not None else None,
        "inference_ms": float(0),  # will be filled below
        "confusion_matrix": cm,
        "num_test_samples": len(y_test),
        "labels": y_test.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
    }
    return results


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def run_model_comparison(
    model_name: str,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    run_cnn: bool = True,
    run_svm: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full comparison for one CNN model.

    Returns a list of result dicts for all evaluated variants (CNN and/or SVM).
    """
    results: List[Dict[str, Any]] = []
    model_info = get_model_info(model_name)
    display_name = model_info.display_name

    # ---- CNN Softmax evaluation ----
    if run_cnn:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating CNN-Softmax: {display_name}")
        logger.info(f"{'='*60}")

        cnn_result = _evaluate_cnn_softmax(model_name, device, X_test, y_test, class_names)
        results.append(cnn_result)

        cm_path = str(RESULTS_CM_DIR / f"{model_name}_cnn_confusion_matrix.png")
        plot_confusion_matrix(
            cnn_result["confusion_matrix"],
            class_names=class_names,
            title=f"{display_name} — CNN Softmax",
            save_path=cm_path,
        )

    # ---- SVM training and evaluation ----
    if run_svm:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training SVM: {display_name}")
        logger.info(f"{'='*60}")

        svm_tag = f"{model_name}_svm"
        clf = SVMClassifier(model_tag=svm_tag)

        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        clf.fit(X_train_val, y_train_val)
        clf.save()

        svm_result = _evaluate_svm_classifier(clf, X_test, y_test, f"{display_name}+SVM", class_names)

        # Measure SVM inference time
        t0 = time.time()
        _ = clf.predict(X_test)
        svm_result["inference_ms"] = (time.time() - t0) / len(y_test) * 1000

        results.append(svm_result)

        cm_path = str(RESULTS_CM_DIR / f"{model_name}_svm_confusion_matrix.png")
        plot_confusion_matrix(
            svm_result["confusion_matrix"],
            class_names=class_names,
            title=f"{display_name} — CNN+SVM",
            save_path=cm_path,
        )

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CNN-Softmax vs CNN+SVM for durian leaf disease detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_models.py                           # All 5 models
  python scripts/compare_models.py --models resnet50         # ResNet50 only
  python scripts/compare_models.py --models resnet50 efficientnet_b0
  python scripts/compare_models.py --no-cnn                  # SVM only
  python scripts/compare_models.py --no-svm                  # CNN only
  python scripts/compare_models.py --force-reextract         # Re-extract features
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to process. Defaults to all 5 models.",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="Skip CNN-Softmax evaluation.",
    )
    parser.add_argument(
        "--no-svm",
        action="store_true",
        help="Skip SVM training and evaluation.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force re-extraction of cached features.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--val-for-train",
        action="store_true",
        help="Include validation set in SVM training (train+val).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_cnn = not args.no_cnn
    run_svm = not args.no_svm

    if not run_cnn and not run_svm:
        logger.error("Must run at least one of --no-cnn or --no-svm (can't skip both).")
        sys.exit(1)

    set_global_seed(RANDOM_STATE)
    device = torch.device(args.device) if args.device else get_device()

    model_names = args.models if args.models else get_all_model_names()
    logger.info(f"Models to process: {model_names}")
    logger.info(f"Device: {device}")
    logger.info(f"Modes: CNN={run_cnn}, SVM={run_svm}")

    dataset_paths = get_dataset_paths()
    logger.info(f"Dataset: {dataset_paths}")

    all_results: List[Dict[str, Any]] = []

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}. Skipping.")
            continue

        model_info = get_model_info(model_name)
        if not model_info.checkpoint_exists():
            logger.warning(f"Checkpoint not found for {model_name}: {model_info.checkpoint_path}")
            logger.warning("Proceeding anyway (will use pretrained weights if available).")

        logger.info(f"\n{'#'*70}")
        logger.info(f"# Model: {model_info.display_name} (feature_dim={model_info.feature_dim})")
        logger.info(f"{'#'*70}")

        # ---- Feature extraction (with caching) ----
        if args.force_reextract:
            for tag in ["train", "val", "test"]:
                cache_file = FEATURE_CACHE_DIR / f"{model_name}_{tag}_features.npy"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Deleted cached features: {cache_file}")

        logger.info("Extracting features...")
        extractor = CNNFeatureExtractor(model_name, device=device)
        X_train, y_train = extractor.extract_from_dataset(dataset_paths["train_dir"], cache_tag="train")
        X_val, y_val = extractor.extract_from_dataset(dataset_paths["val_dir"], cache_tag="val")
        X_test, y_test = extractor.extract_from_dataset(dataset_paths["test_dir"], cache_tag="test")

        logger.info(
            f"Features extracted: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
        )

        # ---- Run comparison for this model ----
        results = run_model_comparison(
            model_name=model_name,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            class_names=DISEASE_CLASSES,
            run_cnn=run_cnn,
            run_svm=run_svm,
        )
        all_results.extend(results)

    # ---- Aggregate results ----
    if not all_results:
        logger.error("No results collected. Exiting.")
        sys.exit(1)

    logger.info(f"\n\n{'#'*70}")
    logger.info(f"# ALL RESULTS SUMMARY")
    logger.info(f"{'#'*70}")
    print_results_table(all_results)

    # ---- Save CSV ----
    csv_path = str(RESULTS_DIR / "metrics_comparison.csv")
    save_results_csv(all_results, save_path=csv_path)

    # ---- Visualization ----
    chart_path = str(RESULTS_DIR / "comparison_chart.png")
    plot_comparison_chart(all_results, save_path=chart_path)

    # ROC curves (only for SVM results with probabilities)
    svm_results = [r for r in all_results if r.get("mode") == "svm"]
    if svm_results:
        plot_roc_curves(all_results, save_path=str(RESULTS_DIR / "roc_curves.png"))

    # ---- Per-model detail ----
    for result in all_results:
        cm = result["confusion_matrix"]
        model_short = result["model_name"].replace(" ", "_").replace("+", "_")
        cm_path = str(RESULTS_CM_DIR / f"{model_short}_cm.png")
        plot_confusion_matrix(
            cm,
            title=f"{result['model_name']} ({result['mode'].upper()})",
            save_path=cm_path,
        )

    logger.info(f"\nAll results saved to: {RESULTS_DIR}")
    logger.info(f"Confusion matrices: {RESULTS_CM_DIR}")
    logger.info(f"SVM models: {SVM_MODELS_DIR}")
    logger.info(f"Feature cache: {FEATURE_CACHE_DIR}")


if __name__ == "__main__":
    main()
