"""
Evaluation metrics and visualizations for PR rejection models.

Given ground-truth labels, predictions, and predicted
probabilities, this module computes standard binary classification metrics, 
persist them as JSON, and generate publication-ready visualization figures.

Covers four evaluation concerns:
- Metric computation and persistence (compute_metrics, save_metrics)
- Per-split summary tables and comparison charts
  (display_metric_table_val/test, plot_metric_bars)
- Diagnostic figure panels (plot_curves_summary, plot_diagnostics_summary,
  plot_split_summary)
- Feature-group ablation study (run_group_ablation)
- Subgroup performance reporting (run_subgroup_report and helpers)

All figures are saved to disk as PDF and not displayed interactively, so this
module works identically in Colab (where plt.show() would be a no-op after a
cell completes) and in local script runs.

Applicability:
  - Core metric and visualization functions (compute_metrics, save_metrics,
    plot_curves_summary, plot_diagnostics_summary, plot_split_summary,
    plot_metric_bars, display_metric_table_val, display_metric_table_test,
    run_subgroup_report): applicable to all models.
  - run_group_ablation: applicable to Multimodal LR only.
"""

import itertools
import json
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Semantic class colors — match final_eda.ipynb palette
ACCEPTED_COLOR = "#357768"   # teal-blue  (label = 0, merged)
REJECTED_COLOR = "#723e46"   # dark maroon (label = 1, closed)

# Split comparison colors (val / test bar charts)
VAL_COLOR  = "#156798"   # dark teal 
TEST_COLOR = "#ea9e2b"   # warm orange

# Categorical palette from final_eda.ipynb — used to colour subgroup bars
CATEGORICAL_PALETTE = [
    "#156798",  # dark teal
    "#ea9e2b",  # warm orange
    "#a45a25",  # rust
    "#939e47",  # forest green
    "#d0a731",  # olive
    "#723e46",  # maroon
    "#357768",  # teal-blue
    "#2290bf",  # medium blue
]

# Human-readable model display names for chart titles
MODEL_DISPLAY_NAMES = {
    "multimodal_lr":        "Multimodal Logistic Regression",
    "multimodal_lr_opt":    "Multimodal Logistic Regression (Optimized Threshold)",
    "qwen3_standard_llm":   "Qwen3 Standard LLM",
    "qwq_reasoning_llm":    "QwQ Reasoning LLM",
}


# Metric computation
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
) -> dict:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        y_prob: Predicted probabilities for class 1, shape (n,).
        split_name: Identifier for the split ('train', 'val', or 'test').
            Included in the returned dict for traceability.

    Returns:
        Dictionary with keys: split, accuracy, precision, recall, f1,
        precision_macro, recall_macro, f1_macro, balanced_accuracy,
        auc_roc, auc_pr.
        precision, recall, f1 are binary metrics for the rejection class
        (label=1, pos_label=1). All metric values are Python floats rounded
        to 4 decimal places.
    """
    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auc_pr": round(float(average_precision_score(y_true, y_prob)), 4),
    }
    logger.info(
        "[%s] accuracy=%.4f  f1=%.4f  auc_roc=%.4f  auc_pr=%.4f  balanced_acc=%.4f",
        split_name,
        metrics["accuracy"],
        metrics["f1"],
        metrics["auc_roc"],
        metrics["auc_pr"],
        metrics["balanced_accuracy"],
    )
    return metrics


# Persistence
def save_metrics(metrics: dict, output_dir: Path, model_name: str = "model") -> None:
    """Append metrics for one split to output_dir/{model_name}_metrics.json.

    If the file already exists it is loaded, the new split entry is
    added (or overwritten if the split key is already present), and the
    file is re-written. This means multiple splits can be accumulated in
    one file across successive calls.

    Args:
        metrics: Dict returned by compute_metrics.
        output_dir: Directory where the metrics JSON is written.
        model_name: Prefix for the output filename (e.g. "multimodal_lr").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{model_name}_metrics.json"

    all_metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)

    split = metrics["split"]
    all_metrics[split] = metrics

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Metrics for split '%s' saved to %s.", split, metrics_path)


def _build_metric_table(metrics: dict, split_label: str) -> pd.DataFrame:
    """Build a single-row metrics DataFrames.

    Args:
        metrics: Dict returned by compute_metrics for one split.
        split_label: Index label for the row (e.g. "Validation" or "Test").

    Returns:
        Single-row DataFrame with columns PR-AUC, ROC-AUC, Precision, Recall,
        F1 (Class 1), F1 Macro, Accuracy.
    """
    row = {
        "PR-AUC":       metrics["auc_pr"],
        "ROC-AUC":      metrics["auc_roc"],
        "Precision":    metrics["precision"],
        "Recall":       metrics["recall"],
        "F1 (Class 1)": metrics["f1"],
        "F1 Macro":     metrics["f1_macro"],
        "Accuracy":     metrics["accuracy"],
    }
    return pd.DataFrame([row], index=[split_label])


def display_metric_table_val(metrics: dict, output_dir: Path, model_name: str = "model") -> pd.DataFrame:
    """Build, print, and save a summary metrics table for the validation split.

    Args:
        metrics: Dict returned by compute_metrics for the validation split.
        output_dir: Directory where the CSV is written.
        model_name: Prefix for the output filename (e.g. "multimodal_lr").

    Returns:
        Single-row DataFrame indexed "Validation" with columns PR-AUC, ROC-AUC,
        Precision, Recall, F1 (Class 1), F1 Macro, Accuracy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _build_metric_table(metrics, "Validation")
    print(df.to_string())
    path = output_dir / f"{model_name}_metric_table_val.csv"
    df.to_csv(path)
    logger.info("Validation metric table saved to %s.", path)
    return df


def display_metric_table_test(metrics: dict, output_dir: Path, model_name: str = "model") -> pd.DataFrame:
    """Build, print, and save a summary metrics table for the test split.

    Args:
        metrics: Dict returned by compute_metrics for the test split.
        output_dir: Directory where the CSV is written.
        model_name: Prefix for the output filename (e.g. "multimodal_lr").

    Returns:
        Single-row DataFrame indexed "Test" with columns PR-AUC, ROC-AUC,
        Precision, Recall, F1 (Class 1), F1 Macro, Accuracy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _build_metric_table(metrics, "Test")
    print(df.to_string())
    path = output_dir / f"{model_name}_metric_table_test.csv"
    df.to_csv(path)
    logger.info("Test metric table saved to %s.", path)
    return df


def plot_metric_bars(
    all_metrics: dict,
    output_dir: Path,
    model_name: str = "model1",
) -> plt.Figure:
    """Save a grouped bar chart comparing key metrics across val and test splits.

    Args:
        all_metrics: Dict keyed by split name ('val', 'test', optionally 'train').
            Each value is a metrics dict returned by compute_metrics.
        output_dir: Directory where the PDF is written.
        model_name: Key into MODEL_DISPLAY_NAMES (e.g. "multimodal_lr"). Falls
            back to the raw string if not found. Default is 'model1'.

    Returns:
        The matplotlib Figure object. The caller is responsible for closing
        it (e.g. with plt.close(fig)) after displaying or discarding it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys   = ["auc_pr", "auc_roc", "precision", "recall", "f1"]
    metric_labels = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1\n(Rejected Class)"]
    splits_to_plot = [s for s in ["val", "test"] if s in all_metrics]
    colors = {"val": VAL_COLOR, "test": TEST_COLOR}

    n_metrics = len(metric_keys)
    n_splits = len(splits_to_plot)
    bar_width = 0.35
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(splits_to_plot):
        values = [all_metrics[split].get(k, 0.0) for k in metric_keys]
        offset = (i - (n_splits - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=split, color=colors[split])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metric Comparison — {display_name} (Val vs Test)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = output_dir / f"{model_name}_metric_comparison.pdf"
    fig.savefig(path)
    logger.info("Metric comparison chart saved to %s.", path)
    return fig


def plot_curves_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    output_dir: Path,
    model_name: str = "model1",
) -> plt.Figure:
    """Save a 1x2 figure with PR-AUC and ROC curves for one split.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels. Not used in this function; accepted to
            keep the signature consistent with plot_diagnostics_summary so
            callers can pass the same arguments to both.
        y_prob: Predicted probabilities for class 1.
        split_name: Used in the figure suptitle and filename.
        output_dir: Directory where the PDF is written.
        model_name: Key into MODEL_DISPLAY_NAMES (e.g. "multimodal_lr"). Falls
            back to the raw string if not found.

    Returns:
        The matplotlib Figure object. The caller is responsible for closing it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{display_name} — {split_name}", fontsize=13, fontweight="bold")

    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, name=display_name, ax=ax_pr
    )
    ax_pr.set_title("Precision-Recall Curve")

    RocCurveDisplay.from_predictions(
        y_true, y_prob, name=display_name, ax=ax_roc
    )
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="No-skill baseline")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    path = output_dir / f"{model_name}_curves_summary_{split_name}.pdf"
    fig.savefig(path)
    logger.info("Curves summary figure saved to %s.", path)
    return fig


def plot_diagnostics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    output_dir: Path,
    model_name: str = "model1",
) -> plt.Figure:
    """Save a 1x2 figure with confusion matrix and probability histogram for one split.

    The histogram uses ACCEPTED_COLOR and REJECTED_COLOR from the EDA palette and
    labels histograms by true outcome to make clear these are ground-truth classes,
    not predictions.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for class 1.
        split_name: Used in the figure suptitle and filename.
        output_dir: Directory where the PDF is written.
        model_name: Key into MODEL_DISPLAY_NAMES (e.g. "multimodal_lr"). Falls
            back to the raw string if not found.

    Returns:
        The matplotlib Figure object. The caller is responsible for closing it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    fig, (ax_cm, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{display_name} — {split_name}", fontsize=13, fontweight="bold")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Accepted (0)", "Rejected (1)"]
    ).plot(ax=ax_cm, colorbar=True, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")

    mask_accepted = y_true == 0
    mask_rejected = y_true == 1
    ax_hist.hist(
        y_prob[mask_accepted],
        bins=30,
        alpha=0.6,
        color=ACCEPTED_COLOR,
        label="True: Accepted",
        density=True,
    )
    ax_hist.hist(
        y_prob[mask_rejected],
        bins=30,
        alpha=0.6,
        color=REJECTED_COLOR,
        label="True: Rejected",
        density=True,
    )
    ax_hist.axvline(
        0.5, color="black", linestyle="--", linewidth=0.9, label="0.5 Threshold"
    )
    ax_hist.set_xlabel("Predicted Rejection Probability")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Predicted Probability Distribution")
    ax_hist.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = output_dir / f"{model_name}_diagnostics_summary_{split_name}.pdf"
    fig.savefig(path)
    logger.info("Diagnostics summary figure saved to %s.", path)
    return fig


def plot_split_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    output_dir: Path,
    model_name: str = "model1",
) -> plt.Figure:
    """Convenience wrapper that calls plot_curves_summary and plot_diagnostics_summary together.

    Saves both the curves summary (PR-AUC + ROC) and diagnostics summary
    (confusion matrix + probability distribution) PDFs for one split in a
    single call, and returns the diagnostics figure.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for class 1.
        split_name: Used in figure suptitles and filenames.
        output_dir: Directory where the PDFs are written.
        model_name: Key into MODEL_DISPLAY_NAMES (e.g. "multimodal_lr"). Falls
            back to the raw string if not found.

    Returns:
        The diagnostics Figure object (from plot_diagnostics_summary).
    """
    plot_curves_summary(y_true, y_pred, y_prob, split_name, output_dir, model_name)
    fig = plot_diagnostics_summary(y_true, y_pred, y_prob, split_name, output_dir, model_name)
    return fig

# Group ablation study
_ABLATION_COLORS = {
    "train": CATEGORICAL_PALETTE[0],  # dark teal
    "val":   CATEGORICAL_PALETTE[1],  # warm orange
    "test":  CATEGORICAL_PALETTE[2],  # rust
}


def _plot_ablation_pr_auc(
    df: pd.DataFrame,
    no_skill_baseline: float,
    eval_dir: Path,
    model_name: str,
) -> plt.Figure:
    """Render and save the grouped bar chart of PR-AUC across ablation subsets.

    Args:
        df: DataFrame returned by run_group_ablation, sorted by val_pr_auc
            descending. Must contain columns: combination, train_pr_auc,
            val_pr_auc, test_pr_auc.
        no_skill_baseline: Proportion of positive (rejected) samples in the
            test set. Drawn as a horizontal dashed reference line.
        eval_dir: Directory where the PNG is written.
        model_name: Prefix for the output filename.

    Returns:
        The matplotlib Figure object. The caller is responsible for closing it.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    combinations = df["combination"].tolist()
    n_combos = len(combinations)
    bar_width = 0.25
    x = np.arange(n_combos)

    fig, ax = plt.subplots(figsize=(max(8, n_combos * 1.4), 5))

    for i, split in enumerate(("train", "val", "test")):
        col = f"{split}_pr_auc"
        values = df[col].tolist()
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=split,
            color=_ABLATION_COLORS[split],
            alpha=0.88,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(f"{display_name} Group Ablation")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = eval_dir / f"{model_name}_ablation_pr_auc.pdf"
    fig.savefig(path)
    logger.info("Ablation PR-AUC chart saved to %s.", path)
    return fig


def run_group_ablation(
    group_slices: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_dir: Path,
    model_name: str = "model1",
    C: float = 1.0,
    random_state: int = 42,
    display_fn: Callable | None = None,
) -> pd.DataFrame:
    """Train a fresh LogisticRegression on every non-empty subset of feature groups.

    Args:
        group_slices: Mapping from feature group name (e.g. "diff", "text",
            "metadata") to a slice object that selects the corresponding
            columns from the feature matrices. Values must be contiguous
            slice objects (as produced by FeaturePipeline.group_slices_).
        X_train: Training feature matrix, shape (n_train, n_features).
        y_train: Training labels, shape (n_train,). Binary: 0=accepted, 1=rejected.
        X_val: Validation feature matrix, shape (n_val, n_features).
        y_val: Validation labels, shape (n_val,).
        X_test: Test feature matrix, shape (n_test, n_features).
        y_test: Test labels, shape (n_test,).
        eval_dir: Directory where the CSV and PDF files are saved.
        model_name: Prefix for all output filenames (e.g. "multimodal_lr").
        C: Inverse regularisation strength for LogisticRegression.
        random_state: Random seed for reproducibility.
        display_fn: Optional callable invoked with each figure after it is
            saved (e.g. IPython.display.display for inline Colab rendering).
            When None, figures are closed immediately after saving.

    Returns:
        DataFrame with one row per non-empty feature-group subset, sorted by
        val_pr_auc descending. Columns: combination, n_features, train_pr_auc,
        val_pr_auc, test_pr_auc (primary metric), train_roc_auc, val_roc_auc,
        test_roc_auc (secondary), train_val_gap (train_pr_auc - val_pr_auc).

    Raises:
        ValueError: If group_slices is empty.
    """
    if not group_slices:
        raise ValueError("group_slices must contain at least one group.")

    # Convert slices to int index lists
    sorted_names = sorted(group_slices.keys())
    group_indices: dict[str, list[int]] = {
        name: list(range(group_slices[name].start, group_slices[name].stop))
        for name in sorted_names
    }

    rows = []
    n_groups = len(sorted_names)
    all_subsets = itertools.chain.from_iterable(
        itertools.combinations(sorted_names, r) for r in range(1, n_groups + 1)
    )

    for subset in all_subsets:
        combination = "+".join(subset)
        idx: list[int] = sorted(
            col for name in subset for col in group_indices[name]
        )

        Xtr = X_train[:, idx]
        Xv = X_val[:, idx]
        Xte = X_test[:, idx]

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=C,
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(Xtr, y_train)

        prob_train = clf.predict_proba(Xtr)[:, 1]
        prob_val = clf.predict_proba(Xv)[:, 1]
        prob_test = clf.predict_proba(Xte)[:, 1]

        train_pr = round(float(average_precision_score(y_train, prob_train)), 4)
        val_pr = round(float(average_precision_score(y_val, prob_val)), 4)
        test_pr = round(float(average_precision_score(y_test, prob_test)), 4)

        rows.append(
            {
                "combination": combination,
                "n_features": len(idx),
                "train_pr_auc": train_pr,
                "val_pr_auc": val_pr,
                "test_pr_auc": test_pr,
                "train_roc_auc": round(float(roc_auc_score(y_train, prob_train)), 4),
                "val_roc_auc": round(float(roc_auc_score(y_val, prob_val)), 4),
                "test_roc_auc": round(float(roc_auc_score(y_test, prob_test)), 4),
                "train_val_gap": round(train_pr - val_pr, 4),
            }
        )
        logger.info(
            "[ablation] %-30s  val_pr_auc=%.4f  test_pr_auc=%.4f  gap=%.4f",
            combination,
            val_pr,
            test_pr,
            train_pr - val_pr,
        )

    df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / f"{model_name}_ablation.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Ablation results saved to %s.", csv_path)

    no_skill_baseline = float((y_test == 1).sum() / len(y_test))

    fig_pr = _plot_ablation_pr_auc(df, no_skill_baseline, eval_dir, model_name)
    if display_fn is not None:
        display_fn(fig_pr)
    plt.close(fig_pr)

    return df


# Subgroup performance reporting
def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    group_dim: str,
    model_name: str,
    split_name: str = "test",
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute binary classification metrics for each subgroup in group_labels.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        y_prob: Predicted probabilities for class 1, shape (n,).
        group_labels: Array of group identifiers, shape (n,). Categorical or string.
        group_dim: Name of the grouping dimension (e.g. "agent_name").
        model_name: Written into the output rows for cross-model stacking.
        split_name: Data split identifier ('train', 'val', or 'test').
        min_n: Minimum group size below which low_support is flagged True.

    Returns:
        DataFrame with one row per unique value in group_labels, sorted by
        pr_auc descending (NaN last). Columns: group_dim, group_value,
        model_name, split, n_samples, rejection_rate, pr_auc, roc_auc,
        precision, recall, f1, low_support.
    """
    rows = []
    for group_val in np.unique(group_labels):
        mask = group_labels == group_val
        n = int(mask.sum())
        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_prob[mask]

        low_support = n < min_n
        if low_support:
            logger.debug(
                "[subgroup] %s=%s has only %d samples (< min_n=%d). "
                "Flagged low_support=True.",
                group_dim, group_val, n, min_n,
            )

        n_classes = len(np.unique(yt))
        if n_classes < 2:
            logger.debug(
                "[subgroup] %s=%s has only one class in y_true. "
                "pr_auc and roc_auc set to NaN.",
                group_dim, group_val,
            )
            pr_auc: float = float("nan")
            roc_auc: float = float("nan")
        else:
            pr_auc = round(float(average_precision_score(yt, yprob)), 4)
            roc_auc = round(float(roc_auc_score(yt, yprob)), 4)

        rows.append({
            "group_dim": group_dim,
            "group_value": group_val,
            "model_name": model_name,
            "split": split_name,
            "n_samples": n,
            "rejection_rate": round(float(yt.mean()), 4),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": round(
                float(precision_score(yt, yp, pos_label=1, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(yt, yp, pos_label=1, zero_division=0)), 4
            ),
            "f1": round(float(f1_score(yt, yp, pos_label=1, zero_division=0)), 4),
            "low_support": low_support,
        })

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def save_subgroup_metrics(
    df: pd.DataFrame,
    eval_dir: Path,
    model_name: str,
    group_dim: str,
) -> Path:
    """Append subgroup metrics to a CSV file, creating it if absent.

    Args:
        df: DataFrame returned by compute_subgroup_metrics.
        eval_dir: Directory where the CSV is written.
        model_name: Prefix for the output filename.
        group_dim: Dimension name, used in the filename suffix.

    Returns:
        Path to the written CSV file.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / f"{model_name}_subgroup_{group_dim}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    logger.info("Subgroup metrics saved to %s.", path)
    return path


def plot_subgroup_pr_auc(
    df: pd.DataFrame,
    eval_dir: Path,
    model_name: str,
    group_dim: str,
    no_skill_baseline: float,
    category_order: list[str] | None = None,
    filter_low_support: bool = False,
    top_n: int | None = None,
) -> plt.Figure:
    """Save a vertical bar chart of PR-AUC per subgroup and return the figure.

    Each bar is coloured by cycling through CATEGORICAL_PALETTE. The no-skill
    baseline is drawn as a horizontal dashed reference line.

    Args:
        df: DataFrame returned by compute_subgroup_metrics. Must contain
            columns: group_value, pr_auc, n_samples, low_support, split.
        eval_dir: Directory where the PNG is written.
        model_name: Used in the filename and figure title.
        group_dim: Grouping dimension name, used in the title and filename.
        no_skill_baseline: Rejection rate of the full split; drawn as a
            horizontal dashed reference line.
        category_order: Optional list of group_value strings specifying display
            order from left to right. When None, bars are sorted by PR-AUC
            descending (highest on the left).
        filter_low_support: When True, rows where low_support=True are removed
            before plotting (useful for high-cardinality dims like language).
        top_n: When set, keeps only the N groups with the largest n_samples
            before plotting. Applied after filter_low_support.

    Returns:
        The matplotlib Figure object. The caller is responsible for closing it.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    if filter_low_support:
        df = df[df["low_support"] == False].copy()

    if top_n is not None:
        df = df.nlargest(top_n, "n_samples").copy()

    if category_order is not None:
        order_map = {v: i for i, v in enumerate(category_order)}
        df_plot = df.copy()
        df_plot["_sort_key"] = df_plot["group_value"].map(order_map).fillna(len(category_order))
        df_plot = df_plot.sort_values("_sort_key", ascending=True).drop(columns="_sort_key")
    else:
        df_plot = df.sort_values("pr_auc", ascending=False, na_position="last").copy()
    split = df["split"].iloc[0]

    labels = [str(row["group_value"]) for _, row in df_plot.iterrows()]

    palette_cycle = list(CATEGORICAL_PALETTE)
    colors = [palette_cycle[i % len(palette_cycle)] for i in range(len(df_plot))]

    fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.9), 5))
    bars = ax.bar(labels, df_plot["pr_auc"], color=colors)

    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        val = row["pr_auc"]
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    valid_vals = df_plot["pr_auc"].dropna()
    max_val = max(valid_vals.max() if not valid_vals.empty else 0.0, no_skill_baseline)
    ax.set_ylim(0, max_val + 0.15)

    ax.axhline(
        no_skill_baseline,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label=f"No-skill baseline ({no_skill_baseline:.4f})",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PR-AUC")
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    ax.set_title(f"{display_name} PR-AUC by {group_dim} — {split.title()}")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = eval_dir / f"{model_name}_subgroup_{group_dim}_{split}.pdf"
    fig.savefig(path)
    logger.info("Subgroup PR-AUC chart saved to %s.", path)
    return fig


def run_subgroup_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_cols: dict[str, np.ndarray],
    eval_dir: Path,
    model_name: str = "model1",
    split_name: str = "test",
    min_n: int = 30,
    category_orders: dict[str, list[str]] | None = None,
    filter_low_support_dims: set[str] | None = None,
    top_n_dims: dict[str, int] | None = None,
    display_fn: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    """Run subgroup metric computation, persistence, and plotting for all dimensions.

    Args:
        y_true: Ground-truth binary labels, shape (n,).
        y_pred: Predicted binary labels, shape (n,).
        y_prob: Predicted probabilities for class 1, shape (n,).
        group_cols: Mapping from dimension name to label array, e.g.
            {"agent_name": agent_arr, "time_period": period_arr}.
        eval_dir: Directory where CSV and PNG files are saved.
        model_name: Written into output filenames and CSV rows.
        split_name: Data split identifier passed to compute_subgroup_metrics.
        min_n: Passed to compute_subgroup_metrics for low-support flagging.
        category_orders: Optional mapping from dimension name to an ordered list
            of group values for chart display (top to bottom). Dimensions not
            present here default to PR-AUC sort order.
        filter_low_support_dims: Set of dimension names for which low-support
            groups are excluded from the chart (but kept in the CSV).
        top_n_dims: Optional mapping from dimension name to an integer N. When
            present for a dimension, the chart shows only the N groups with the
            largest sample counts (applied after filter_low_support).
        display_fn: Optional callable invoked with each figure and styled
            DataFrame after saving (e.g. IPython.display.display for inline
            Colab rendering). When None, figures are closed without display.

    Returns:
        Dict mapping each dimension name to its computed metrics DataFrame.
    """
    no_skill_baseline = float((y_true == 1).sum() / len(y_true))
    results: dict[str, pd.DataFrame] = {}

    for dim, labels in group_cols.items():
        df = compute_subgroup_metrics(
            y_true, y_pred, y_prob, labels, dim, model_name, split_name, min_n
        )
        save_subgroup_metrics(df, eval_dir, model_name, dim)
        order = (category_orders or {}).get(dim)
        fls = dim in (filter_low_support_dims or set())
        tn = (top_n_dims or {}).get(dim)
        fig = plot_subgroup_pr_auc(
            df, eval_dir, model_name, dim, no_skill_baseline,
            category_order=order, filter_low_support=fls, top_n=tn,
        )
        if display_fn is not None:
            display_fn(fig)
        plt.close(fig)

        if display_fn is not None:
            _display_cols = [
                "group_value", "n_samples", "rejection_rate",
                "pr_auc", "roc_auc", "precision", "recall", "f1", "low_support",
            ]
            _styled = (
                df[_display_cols]
                .style
                .set_caption(f"{dim} — {split_name}")
                .format(
                    {
                        "rejection_rate": "{:.3f}",
                        "pr_auc":         "{:.4f}",
                        "roc_auc":        "{:.4f}",
                        "precision":      "{:.4f}",
                        "recall":         "{:.4f}",
                        "f1":             "{:.4f}",
                    },
                    na_rep="—",
                )
                .hide(axis="index")
            )
            display_fn(_styled)

        valid = df["pr_auc"].dropna()
        best = df.loc[df["pr_auc"].idxmax(), "group_value"] if not valid.empty else "N/A"
        worst = df.loc[df["pr_auc"].idxmin(), "group_value"] if not valid.empty else "N/A"
        logger.info(
            "[subgroup report] dim=%s  n_groups=%d  no_skill=%.4f  "
            "best=%s (%.4f)  worst=%s (%.4f)",
            dim,
            len(df),
            no_skill_baseline,
            best,
            valid.max() if not valid.empty else float("nan"),
            worst,
            valid.min() if not valid.empty else float("nan"),
        )
        results[dim] = df

    return results
