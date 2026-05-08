"""Evaluation and visualization for CNN+SVM comparison.

Provides:
  - evaluate_model(): compute all required metrics for one model
  - plot_confusion_matrix(): heatmap PNG
  - plot_comparison_chart(): grouped bar chart of all metrics
  - plot_roc_curves(): multiclass ROC curves (One-vs-Rest)
  - plot_feature_distribution(): PCA/t-SNE of extracted features (bonus)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DISEASE_CLASSES, DISEASE_CLASSES_VI, RESULTS_CM_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_names: Optional[List[str]] = None,
    mode: str = "svm",
) -> Dict[str, Any]:
    """Evaluate a classifier and compute all required metrics.

    Args:
        model: Fitted classifier with a `predict()` method.
               If mode="svm", also expects `predict_proba()`.
        X_test: Test feature matrix (n_samples, n_features).
        y_test: True test labels (n_samples,).
        model_name: Display name for this model (e.g. "ResNet50+SVM").
        class_names: List of class names. Defaults to the 5 disease classes.
        mode: Either "svm" (has predict_proba) or "cnn" (softmax only).

    Returns:
        Dict with: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
        roc_auc, inference_ms (per sample), confusion_matrix, model_name, mode.
    """
    if class_names is None:
        class_names = DISEASE_CLASSES

    n_classes = len(class_names)

    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_ns = time.time() - t0
    inference_ms_per_sample = (inference_ns / len(y_test)) * 1000

    preds = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
    labels = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)

    roc_auc = None
    if mode == "svm" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 1:
                y_proba = np.column_stack([1 - y_proba[:, 0], y_proba[:, 0]])
            roc_auc = roc_auc_score(
                labels, y_proba, multi_class="ovr", average="macro", labels=list(range(n_classes))
            )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")
            roc_auc = None

    results = {
        "model_name": model_name,
        "mode": mode,
        "accuracy": float(acc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "inference_ms": float(inference_ms_per_sample),
        "confusion_matrix": cm,
        "num_test_samples": len(y_test),
        "classification_report": classification_report(
            labels, preds, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    logger.info(
        f"[{model_name}] Acc={acc:.4f} | F1-macro={f1_m:.4f} | "
        f"Precision-macro={precision_m:.4f} | Recall-macro={recall_m:.4f}"
        + (f" | ROC-AUC={roc_auc:.4f}" if roc_auc is not None else "")
        + f" | {inference_ms_per_sample:.3f}ms/sample"
    )

    return results


def _get_short_label(class_name: str) -> str:
    """Short display label for a class."""
    return DISEASE_CLASSES_VI.get(class_name, class_name)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = False,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix as numpy array.
        class_names: List of class names.
        title: Plot title.
        save_path: If provided, saves PNG to this path.
        normalize: If True, show percentages instead of counts.
        cmap: seaborn colormap name.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if class_names is None:
        class_names = DISEASE_CLASSES

    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        cm_norm = cm
        fmt = "d"

    short_names = [_get_short_label(c) for c in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"label": "Ty le" if normalize else "So luong"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Du doan", fontsize=12)
    ax.set_ylabel("Thuc te", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title("Ma tran nham lan (Confusion Matrix)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_comparison_chart(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Create a grouped bar chart comparing all models across multiple metrics.

    Args:
        results_list: List of result dicts from evaluate_model().
        save_path: If provided, saves PNG to this path.

    Returns:
        matplotlib Figure object.
    """
    metric_names = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    display_names = ["Accuracy", "F1-Macro", "F1-Weighted", "Precision", "Recall"]

    df = pd.DataFrame(results_list)
    model_names = df["model_name"].tolist()

    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_models)
    bar_width = 0.14
    multiplier = 0

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, (metric, disp) in enumerate(zip(metric_names, display_names)):
        values = [df.loc[df["model_name"] == name, metric].values[0] for name in model_names]
        offset = bar_width * multiplier
        bars = ax.bar(x + offset, values, bar_width, label=disp, color=colors[i], edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=0)
        multiplier += 1

    ax.set_xlabel("Mo hinh", fontsize=12)
    ax.set_ylabel("Diem so", fontsize=12)
    ax.set_title("So sanh CNN vs CNN+SVM — Cac chi so hieu suat", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.legend(loc="lower right", ncol=n_metrics, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison chart saved to {save_path}")

    return fig


def plot_roc_curves(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot multiclass ROC curves (One-vs-Rest) for all SVM models.

    Args:
        results_list: List of result dicts. Must include 'probabilities' key.
        save_path: If provided, saves PNG.

    Returns:
        matplotlib Figure object.
    """
    from sklearn.metrics import roc_curve, auc

    if class_names is None:
        class_names = DISEASE_CLASSES

    class_names = DISEASE_CLASSES
    short_names = [_get_short_label(c) for c in class_names]
    n_classes = len(class_names)

    svm_results = [r for r in results_list if r.get("mode") == "svm" and "probabilities" in r]

    if not svm_results:
        logger.warning("No SVM results with probabilities found for ROC curves.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Khong co du lieu SVM de ve ROC", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    colors = sns.color_palette("husl", n_classes)

    for r_idx, result in enumerate(svm_results):
        model_name = result["model_name"]
        probabilities = result.get("probabilities")
        y_true = result.get("labels")

        if probabilities is None or y_true is None:
            continue

        for cls_idx in range(n_classes):
            y_true_binary = (np.array(y_true) == cls_idx).astype(int)
            proba_positive = np.array(probabilities)[:, cls_idx]
            fpr, tpr, _ = roc_curve(y_true_binary, proba_positive)
            roc_auc = auc(fpr, tpr)

            lw = 2 if cls_idx == 0 else 1.2
            ls = linestyles[r_idx % len(linestyles)]
            label = f"{model_name} vs {_get_short_label(class_names[cls_idx])} (AUC={roc_auc:.3f})"
            ax.plot(fpr, tpr, color=colors[cls_idx], linestyle=ls, linewidth=lw, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ngau nhien")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Ty le duong tinh gia (FPR)", fontsize=12)
    ax.set_ylabel("Ty le duong tinh that (TPR)", fontsize=12)
    ax.set_title("Duong cong ROC — Phan loai da lop (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curves saved to {save_path}")

    return fig


def _format_results_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format a list of result dicts into a clean DataFrame."""
    rows = []
    for r in results_list:
        row = {
            "Model": r["model_name"],
            "Mode": r["mode"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "F1-Macro": f"{r['f1_macro']:.4f}",
            "F1-Weighted": f"{r['f1_weighted']:.4f}",
            "Precision": f"{r['precision_macro']:.4f}",
            "Recall": f"{r['recall_macro']:.4f}",
            "ROC-AUC": f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A",
            "Infer(ms)": f"{r['inference_ms']:.3f}",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_results_table(results_list: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to console."""
    df = _format_results_table(results_list)
    print("\n" + "=" * 120)
    print("KET QUA SO SANH — CNN vs CNN+SVM")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")


def save_results_csv(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Save results to CSV, including the summary table and per-model metrics."""
    if save_path is None:
        save_path = str(RESULTS_DIR / "metrics_comparison.csv")

    df = _format_results_table(results_list)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results CSV saved to {save_path}")
    return df
