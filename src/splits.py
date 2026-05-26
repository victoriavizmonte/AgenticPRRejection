"""
Temporal train/validation/test split for Agentic PR Rejection.

Given a preprocessed DataFrame with binary labels,
this module produces three chronologically ordered splits that respect temporal
causality (no future information leaks into past).

Split strategy: Percentile-based temporal split.
  - Sort all PRs by created_at timestamp
  - Train: positions 0 to train_end (earliest 70% by index)
  - Val:   positions train_end to val_end (next 15% by index,
           ceiling applied so extra PR goes to val not test)
  - Test:  positions val_end to end (remaining ~15% by index)

  Boundary dates reported in logs are descriptive results of
  where split indices fall in the sorted data, not hard date
  cutoffs applied during splitting.

This ensures adequate sample sizes while maintaining strict temporal
ordering, which is critical for evaluating generalization to future
deployment conditions.

Note: Distribution shift over time (e.g., changing rejection rates)
is real and should be reported, not hidden. The test set represents
the most recent data and may have different characteristics than train.

Applicability:
  - Applicable to all models. All model notebooks use the same temporal
    split produced by this module.
"""

import logging
import math
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    timestamp_col: str = "created_at",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test using temporal percentiles.

    PRs are sorted by creation timestamp and split at the 70th and
    85th percentile positions by index (default). The boundary dates
    reported in logs are descriptive results of where those indices
    fall in the sorted data, not hard date cutoffs. PRs sharing an
    identical creation timestamp at a split boundary are assigned to
    splits by their position in the sorted order.

    Args:
        df: Preprocessed DataFrame with binary labels and timestamps.
            Must contain columns: {timestamp_col}, 'label'.
        train_pct: Fraction of data for training (default: 0.70).
        val_pct: Fraction of data for validation (default: 0.15).
            Test fraction is implicitly 1 - train_pct - val_pct.
            math.ceil is applied to the val index so that when total
            PRs do not divide evenly, the extra PR goes to val rather
            than test.
        timestamp_col: Name of the timestamp column for sorting.
            Default is 'created_at'.

    Returns:
        Tuple of (train_df, val_df, test_df), each a DataFrame with
        the same schema as the input. For n=10,648 PRs with default
        ratios, this yields train=7,453 / val=1,598 / test=1,597.

    Raises:
        ValueError: If train_pct + val_pct >= 1.0, or if required
            columns are missing.

    Example:
        >>> train, val, test = temporal_split(pr_df_labeled)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    # Validate inputs
    if train_pct + val_pct >= 1.0:
        raise ValueError(
            f"train_pct ({train_pct}) + val_pct ({val_pct}) must be < 1.0"
        )

    for col in (timestamp_col, "label"):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    # Convert timestamp to datetime if needed
    if df[timestamp_col].dtype == "object":
        logger.warning(
            f"Column '{timestamp_col}' is string type. Converting to datetime."
        )
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort by timestamp
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n_total = len(df_sorted)

    # Compute split indices; ceiling val to ensure val >= test when sizes differ
    train_end = int(n_total * train_pct)
    val_end = train_end + math.ceil(n_total * val_pct)

    # Split
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # Log split statistics
    logger.info("Temporal split complete (percentile-based):")
    logger.info(f"  Train: {len(train_df)} PRs ({100*len(train_df)/n_total:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} PRs ({100*len(val_df)/n_total:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} PRs ({100*len(test_df)/n_total:.1f}%)")

    logger.info("Date ranges:")
    logger.info(
        f"  Train: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}"
    )
    logger.info(
        f"  Val:   {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}"
    )
    logger.info(
        f"  Test:  {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}"
    )

    # Log label distributions
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_rejected = split_df["label"].sum()
        rej_rate = n_rejected / len(split_df) if len(split_df) > 0 else 0
        logger.info(f"  {name} rejection rate: {rej_rate:.3f}")

    return train_df, val_df, test_df
