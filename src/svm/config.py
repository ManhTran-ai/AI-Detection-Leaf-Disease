"""Centralized configuration for CNN Feature Extraction + SVM Classification pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ── PROJECT_ROOT: support Kaggle import notebook via environment variable ──────
# Cell 1 notebook sets AI_PROJECT_ROOT before importing this module.
# Priority: (1) env var  (2) scan /kaggle/input/  (3) relative to this file
def _resolve_project_root() -> Path:
    # 1) Env var (set by notebook Cell 1)
    if "AI_PROJECT_ROOT" in os.environ:
        p = Path(os.environ["AI_PROJECT_ROOT"])
        if p.is_dir():
            return p

    # 2) Scan /kaggle/input/ (supports any nesting depth)
    if os.path.exists("/kaggle"):
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            for _dir in input_dir.rglob("src"):          # find any dir containing src/
                proj = _dir.parent
                if (proj / "configs").is_dir():
                    return proj
            # fallback: first dir that has src/
            for _dir in input_dir.rglob("src"):
                return _dir.parent

    # 3) Relative to this file (local development)
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = _resolve_project_root()

DISEASE_CLASSES: List[str] = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK",
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT",
]

DISEASE_CLASSES_VI: Dict[str, str] = {
    "ALGAL_LEAF_SPOT": "Ben dot tao",
    "ALLOCARIDARA_ATTACK": "Bo tri tan cong",
    "HEALTHY_LEAF": "La khoe manh",
    "LEAF_BLIGHT": "Benh chay la",
    "PHOMOPSIS_LEAF_SPOT": "Benh dom la Phomopsis",
}


@dataclass
class ModelInfo:
    model_name: str
    display_name: str
    checkpoint_path: str
    config_path: str
    feature_dim: int
    image_size: int
    classifier_attr: str
    family: str

    def checkpoint_exists(self) -> bool:
        return Path(self.checkpoint_path).exists()


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "resnet18": ModelInfo(
        model_name="resnet18",
        display_name="ResNet18",
        checkpoint_path=str(PROJECT_ROOT / "models/checkpoints/resnet18/best_model.pth"),
        config_path=str(PROJECT_ROOT / "configs/config_resnet18.yaml"),
        feature_dim=512,
        image_size=224,
        classifier_attr="fc",
        family="resnet",
    ),
    "resnet34": ModelInfo(
        model_name="resnet34",
        display_name="ResNet34",
        checkpoint_path=str(PROJECT_ROOT / "models/checkpoints/resnet34/best_model.pth"),
        config_path=str(PROJECT_ROOT / "configs/config_resnet34.yaml"),
        feature_dim=512,
        image_size=224,
        classifier_attr="fc",
        family="resnet",
    ),
    "resnet50": ModelInfo(
        model_name="resnet50",
        display_name="ResNet50",
        checkpoint_path=str(PROJECT_ROOT / "models/checkpoints/resnet50/best_model.pth"),
        config_path=str(PROJECT_ROOT / "configs/config_resnet50.yaml"),
        feature_dim=2048,
        image_size=256,
        classifier_attr="fc",
        family="resnet",
    ),
    "mobilenetv3_large": ModelInfo(
        model_name="mobilenetv3_large",
        display_name="MobileNetV3-Large",
        checkpoint_path=str(PROJECT_ROOT / "models/checkpoints/mobilenetv3_large/best_model.pth"),
        config_path=str(PROJECT_ROOT / "configs/config_mobilenetv3.yaml"),
        feature_dim=960,
        image_size=224,
        classifier_attr="classifier",
        family="mobilenet",
    ),
    "efficientnet_b0": ModelInfo(
        model_name="efficientnet_b0",
        display_name="EfficientNet-B0",
        checkpoint_path=str(PROJECT_ROOT / "models/checkpoints/efficientnet_b0/best_model.pth"),
        config_path=str(PROJECT_ROOT / "configs/config_efficientnet_b0.yaml"),
        feature_dim=1280,
        image_size=224,
        classifier_attr="classifier",
        family="efficientnet",
    ),
}


@dataclass
class SVMSearchSpace:
    C: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    gamma: List[str] = field(default_factory=lambda: ["scale", "auto"])
    kernel: List[str] = field(default_factory=lambda: ["rbf"])
    scoring: str = "f1_macro"
    cv_folds: int = 5
    n_jobs: int = -1


# ── OUTPUT_ROOT: writable directory (Kaggle /kaggle/working vs local project root) ──
# On Kaggle, /kaggle/input/ is read-only, so ALL output goes to /kaggle/working/
# This MUST be resolved BEFORE defining output directory variables below.
def _resolve_output_root() -> Path:
    if os.path.exists("/kaggle"):
        kw = Path("/kaggle/working")
        try:
            (kw / ".write_test").touch()
            (kw / ".write_test").unlink()
            return kw
        except OSError:
            pass
        # Fallback: /tmp
        fallback = Path("/tmp/kaggle_output")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return PROJECT_ROOT

OUTPUT_ROOT = _resolve_output_root()

DATASET_CONFIG: Dict[str, Dict[str, str]] = {
    "local": {
        "train_dir": str(PROJECT_ROOT / "data/processed/train"),
        "val_dir":   str(PROJECT_ROOT / "data/processed/val"),
        "test_dir":  str(PROJECT_ROOT / "data/processed/test"),
    },
    "kaggle": {
        # Read from dataset input; auto-detected at runtime
        "train_dir": "AUTO",
        "val_dir":   "AUTO",
        "test_dir":  "AUTO",
    },
}

FEATURE_CACHE_DIR = OUTPUT_ROOT / "features"
RESULTS_DIR       = OUTPUT_ROOT / "results" / "svm_comparison"
RESULTS_CM_DIR    = RESULTS_DIR / "confusion_matrices"
RESULTS_ROC_DIR   = RESULTS_DIR / "roc_curves"
SVM_MODELS_DIR    = OUTPUT_ROOT / "models" / "svm_models"

SVM_SEARCH_SPACE = SVMSearchSpace()
RANDOM_STATE: int = 42

IMAGE_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGE_STD: List[float] = [0.229, 0.224, 0.225]
IMG_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

BATCH_SIZE: int = 32

for _dir in [FEATURE_CACHE_DIR, RESULTS_DIR, RESULTS_CM_DIR, RESULTS_ROC_DIR, SVM_MODELS_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)


def get_dataset_env() -> str:
    """Detect whether running on Kaggle or local environment."""
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def get_dataset_paths() -> Dict[str, str]:
    """Return dataset paths based on current environment.

    On Kaggle, paths are auto-detected by the notebook's Cell 1 and
    returned via the global TRAIN_DIR / VAL_DIR / TEST_DIR variables.
    """
    env = get_dataset_env()
    cfg = DATASET_CONFIG.get(env, DATASET_CONFIG["local"])
    if env == "kaggle":
        # Notebook sets global TRAIN_DIR/VAL_DIR/TEST_DIR after auto-detection;
        # this function is a fallback for scripts.
        global TRAIN_DIR, VAL_DIR, TEST_DIR
        for key in ("train_dir", "val_dir", "test_dir"):
            if cfg.get(key) == "AUTO":
                cfg[key] = str(OUTPUT_ROOT / key.replace("_dir", ""))
    return cfg


def get_model_info(model_name: str) -> ModelInfo:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_all_model_names() -> List[str]:
    return list(MODEL_REGISTRY.keys())
