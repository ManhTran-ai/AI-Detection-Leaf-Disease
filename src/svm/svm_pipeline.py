"""SVM Classifier pipeline with GridSearchCV hyperparameter tuning.

Wraps StandardScaler + SVC in a sklearn Pipeline, tunes hyperparameters
using StratifiedKFold cross-validation, and caches the best model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_STATE, SVM_MODELS_DIR, SVM_SEARCH_SPACE, SVMSearchSpace

logger = logging.getLogger(__name__)


class SVMClassifier:
    """SVM classifier with hyperparameter tuning via GridSearchCV.

    The classifier is wrapped in a sklearn Pipeline: StandardScaler -> SVC.
    GridSearchCV finds the best hyperparameters on the training set, then
    the best model is re-fitted on the full training set.

    Usage:
        clf = SVMClassifier(model_tag="resnet50_svm")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

    Attributes:
        model_tag: Short identifier used for saving/loading the model.
        best_params_: Best hyperparameters found by GridSearchCV.
        best_cv_score_: Best cross-validation score (F1 macro).
        pipeline_: The fitted sklearn Pipeline.
    """

    def __init__(
        self,
        model_tag: str,
        search_space: Optional[SVMSearchSpace] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model_tag: Identifier for saving the model (e.g. "resnet50_svm").
            search_space: Hyperparameter search space. Defaults to SVM_SEARCH_SPACE.
            verbose: Whether to log progress.
        """
        self.model_tag = model_tag
        self.search_space = search_space or SVM_SEARCH_SPACE
        self.verbose = verbose
        self._pipeline: Optional[Pipeline] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_cv_score: Optional[float] = None
        self._cv_results: Optional[Dict[str, Any]] = None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("Classifier not fitted yet. Call fit() first.")
        return self._pipeline

    @property
    def best_params(self) -> Dict[str, Any]:
        if self._best_params is None:
            raise RuntimeError("Classifier not fitted yet.")
        return self._best_params

    @property
    def best_cv_score(self) -> float:
        if self._best_cv_score is None:
            raise RuntimeError("Classifier not fitted yet.")
        return self._best_cv_score

    def _build_pipeline(self) -> Pipeline:
        """Build the scaler + SVM pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ])

    def _build_param_grid(self) -> Dict[str, List[Any]]:
        """Build GridSearchCV parameter grid from search space."""
        grid: Dict[str, List[Any]] = {}
        if self.search_space.C:
            grid["svm__C"] = self.search_space.C
        if self.search_space.gamma:
            grid["svm__gamma"] = self.search_space.gamma
        if self.search_space.kernel:
            grid["svm__kernel"] = self.search_space.kernel
        return grid

    def _build_cv(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.search_space.cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Tune hyperparameters with GridSearchCV and fit on full training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self for chaining.
        """
        if self.verbose:
            logger.info(f"[{self.model_tag}] Starting SVM training on X={X.shape}, y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        pipeline = self._build_pipeline()
        param_grid = self._build_param_grid()
        cv = self._build_cv()

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.search_space.scoring,
            cv=cv,
            n_jobs=self.search_space.n_jobs,
            refit=True,
            verbose=1 if self.verbose else 0,
            return_train_score=True,
        )

        t0 = time.time()
        grid_search.fit(X, y)
        elapsed = time.time() - t0

        self._pipeline = grid_search.best_estimator_
        self._best_params = grid_search.best_params_
        self._best_cv_score = grid_search.best_score_
        self._cv_results = grid_search.cv_results_

        logger.info(
            f"[{self.model_tag}] GridSearchCV complete in {elapsed:.1f}s. "
            f"Best params: {self._best_params}, "
            f"Best CV F1-macro: {self._best_cv_score:.4f}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels as 1D array.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Requires probability=True in SVC (always set here).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        return self.pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, scoring: str = "f1_macro") -> float:
        """Compute a score on given data.

        Args:
            X: Feature matrix.
            y: True labels.
            scoring: Metric name ("f1_macro", "accuracy", etc.).

        Returns:
            Score value.
        """
        from sklearn.metrics import get_scorer
        return get_scorer(scoring)(self.pipeline, X, y)

    def save(self, path: Optional[str] = None) -> str:
        """Save the best model to disk using joblib.

        Args:
            path: Custom save path. If None, saves to SVM_MODELS_DIR/{model_tag}.joblib.

        Returns:
            Path where the model was saved.
        """
        if self._pipeline is None:
            raise RuntimeError("No model to save. Call fit() first.")

        if path is None:
            Path(SVM_MODELS_DIR).mkdir(parents=True, exist_ok=True)
            path = str(SVM_MODELS_DIR / f"{self.model_tag}.joblib")

        joblib.dump(self.pipeline, path)
        logger.info(f"[{self.model_tag}] Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str, model_tag: str, verbose: bool = True) -> "SVMClassifier":
        """Load a previously saved model.

        Args:
            path: Path to the saved .joblib file.
            model_tag: Identifier for the loaded model.
            verbose: Whether to log.

        Returns:
            Loaded SVMClassifier instance.
        """
        instance = cls(model_tag=model_tag, verbose=verbose)
        instance._pipeline = joblib.load(path)
        logger.info(f"[{model_tag}] Model loaded from {path}")
        return instance

    def get_cv_summary(self) -> Dict[str, Any]:
        """Return a summary dict of cross-validation results."""
        if self._cv_results is None:
            raise RuntimeError("Classifier not fitted yet.")
        return {
            "model_tag": self.model_tag,
            "best_params": self.best_params,
            "best_cv_score": self.best_cv_score,
            "cv_folds": self.search_space.cv_folds,
            "scoring": self.search_space.scoring,
            "param_grid": self._build_param_grid(),
            "cv_results": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in self._cv_results.items()
            },
        }
