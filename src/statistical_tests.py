"""Bootstrap statistical tests for PR-AUC evaluation.

This module provides two functions:
  - bootstrap_pr_auc_ci: confidence interval for a single model's PR-AUC.
  - bootstrap_pairwise_test: paired significance test for the difference
    in PR-AUC between two models on the same test set.

All randomness is controlled via np.random.default_rng(seed).

Applicability:
  - Applicable to all models. Used in cross-model significance testing
    and confidence interval estimation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute a bootstrap confidence interval for PR-AUC (average precision).

    Resamples the evaluation set with replacement n_bootstrap times and
    computes sklearn's average_precision_score (pos_label=1) on each
    resample. The confidence interval is the percentile interval of the
    resulting bootstrap distribution.

    Args:
        y_true: Ground-truth binary labels (0 = accepted, 1 = rejected).
            Shape (n_samples,).
        y_score: Predicted probability scores for the positive class.
            Shape (n_samples,).
        n_bootstrap: Number of bootstrap resamples. Default 10_000.
        alpha: Significance level. The returned interval covers
            1 - alpha of the bootstrap distribution. Default 0.05.
        seed: Seed for np.random.default_rng. Guarantees reproducibility
            without touching global random state. Default 42.

    Returns:
        A dict with keys:
            pr_auc (float): Observed PR-AUC on the original, unresampled data.
            ci_lower (float): alpha/2 percentile of the bootstrap distribution.
            ci_upper (float): (1 - alpha/2) percentile of the bootstrap
                distribution.
            n_bootstrap (int): Number of resamples used.
            alpha (float): Significance level used.

    Raises:
        ValueError: If y_true and y_score have different lengths.
        ValueError: If y_true contains fewer than 2 unique class labels.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score must have the same length, "
            f"got {len(y_true)} and {len(y_score)}."
        )
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain at least 2 unique classes to compute PR-AUC. "
            f"Found classes: {np.unique(y_true).tolist()}."
        )

    n = len(y_true)
    rng = np.random.default_rng(seed)

    observed = float(average_precision_score(y_true, y_score, pos_label=1))

    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = average_precision_score(
            y_true[idx], y_score[idx], pos_label=1
        )

    ci_lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    return {
        "pr_auc": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def bootstrap_pairwise_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test for the difference in PR-AUC.

    Compares two models evaluated on the same test set. The observed
    difference is PR-AUC(A) - PR-AUC(B) on the original data. Each
    bootstrap iteration applies the same resample indices to all three
    arrays and recomputes the difference. The two-tailed p-value uses
    the shift method: the proportion of bootstrap differences whose
    absolute deviation from the observed difference is at least as large
    as the observed difference itself.

    Args:
        y_true: Ground-truth binary labels (0 = accepted, 1 = rejected).
            Shape (n_samples,).
        scores_a: Predicted probability scores for model A. Shape (n_samples,).
        scores_b: Predicted probability scores for model B. Shape (n_samples,).
        n_bootstrap: Number of bootstrap resamples. Default 10_000.
        alpha: Significance level for the confidence interval and the
            `significant` flag. Default 0.05.
        seed: Seed for np.random.default_rng. Default 42.

    Returns:
        A dict with keys:
            observed_diff (float): PR-AUC(A) - PR-AUC(B) on original data.
            ci_lower (float): alpha/2 percentile of the bootstrap diff
                distribution.
            ci_upper (float): (1 - alpha/2) percentile of the bootstrap diff
                distribution.
            p_value (float): Two-tailed p-value from the shift method.
            significant (bool): True if p_value < alpha.
            n_bootstrap (int): Number of resamples used.

    Raises:
        ValueError: If y_true, scores_a, and scores_b do not all have the
            same length.
        ValueError: If y_true contains fewer than 2 unique class labels.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError(
            "y_true, scores_a, and scores_b must all have the same length, "
            f"got {len(y_true)}, {len(scores_a)}, {len(scores_b)}."
        )
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain at least 2 unique classes to compute PR-AUC. "
            f"Found classes: {np.unique(y_true).tolist()}."
        )

    n = len(y_true)
    rng = np.random.default_rng(seed)

    observed_diff = float(
        average_precision_score(y_true, scores_a, pos_label=1)
        - average_precision_score(y_true, scores_b, pos_label=1)
    )

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diff_i = average_precision_score(
            y_true[idx], scores_a[idx], pos_label=1
        ) - average_precision_score(y_true[idx], scores_b[idx], pos_label=1)
        boot_diffs[i] = diff_i

    # Two-tailed shift method: centre the bootstrap distribution at zero,
    # then count how often the shifted statistic is at least as extreme.
    p_value = float(
        np.mean(np.abs(boot_diffs - observed_diff) >= np.abs(observed_diff))
    )

    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
    }
