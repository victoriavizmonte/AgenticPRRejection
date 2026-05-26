"""
Logistic Regression training for Multimodal LR (Model 1).

This module fits a Logistic Regression classifier on pre-extracted
CodeBERT embeddings, and save/load the resulting model to disk.

Design decisions:
  - class_weight='balanced': compensates for the approx 60/40 class distribution so that
    the minority class (rejected PRs) is not under-penalized.
  - solver='lbfgs': well-suited for a dense 768-dim feature space with a
    moderate number of samples (~6,000 training PRs).
  - No feature scaling applied: CodeBERT [CLS] embeddings are already in a
    comparable range across dimensions. Scaling did not improve results in
    preliminary runs and adds a fitting-on-train-only step that adds
    complexity without benefit here.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from config import LR_C, LR_MAX_ITER, RANDOM_SEED

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = LR_C,
    max_iter: int = LR_MAX_ITER,
    seed: int = RANDOM_SEED,
) -> LogisticRegression:
    """Fit a Logistic Regression on training embeddings.

    Args:
        X_train: Training embedding matrix, shape (n_train, embedding_dim).
        y_train: Binary labels, shape (n_train,). 0=accepted, 1=rejected.
        C: Inverse regularization strength. Smaller values mean stronger
            regularization. Default from config (LR_C=1.0).
        max_iter: Maximum number of solver iterations. Default from config
            (LR_MAX_ITER=5000).
        seed: Random seed for reproducibility.

    Returns:
        Fitted LogisticRegression instance.

    Raises:
        ValueError: If X_train and y_train have mismatched lengths.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} elements."
        )

    logger.info(
        "Training Logistic Regression: n_train=%d, C=%.4f, max_iter=%d, seed=%d",
        len(X_train),
        C,
        max_iter,
        seed,
    )

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    if model.n_iter_[0] >= max_iter:
        logger.warning(
            "LR did not converge within %d iterations. "
            "Consider increasing LR_MAX_ITER in config.py.",
            max_iter,
        )
    else:
        logger.info(
            "LR converged in %d iterations (%.1fs).", model.n_iter_[0], elapsed
        )

    return model


def save_model(model: LogisticRegression, output_path: Path) -> None:
    """Save a trained Logistic Regression model to disk using joblib.

    Args:
        model: Fitted LogisticRegression instance.
        output_path: Full path (including filename) where the model is saved.
            The parent directory is created if it does not exist.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved to %s.", output_path)


