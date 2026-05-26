"""
Feature extraction for Agentic PR Rejection prediction.

Given a split DataFrame (train/val/test), this module produces
feature representations for each PR. This module must never modify
labels or split assignments.

Chunking is performed after the temporal split so that no
statistics-derived decisions (e.g., percentile-based thresholds)
can leak test-set information into preprocessing.

Applicability:
  - Applicable to Multimodal LR only. Standard LLM and reasoning LLM classifiers
    format raw diff text directly in their own prompt-formatting functions and
    do not import from this module.

Chunking design decisions:
  - Input : pr_commit_details rows joined to a split DataFrame
  - Output: list of chunk strings per PR (ready for CodeBERT tokenization)
  - Strategy : hunk-based (split on @@ markers), first-K selection
  - Default K: 5 (fixed by architecture, not data-derived)
  - Fallback : if no @@ markers are found, truncate raw patch to max_chars
  - total_changes is returned as a scalar feature alongside chunk strings,
    because EDA and related literature confirmed it carries independent predictive signal.
"""

import re
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Constants

# CodeBERT tokenizer limit is 512 tokens. As a character-level proxy before
# tokenization, 1800 chars ≈ 400-500 tokens for typical diff text, giving
# headroom for the [CLS]/[SEP] special tokens. Adjust if profiling shows
# consistent truncation warnings from the tokenizer.
MAX_CHARS_PER_CHUNK: int = 1800

# Number of hunks to keep per PR. Fixed by architecture choice.
DEFAULT_TOP_K_HUNKS: int = 5

# Minimum hunk length in characters for encoding. Filters out
# @@ header-only hunks that carry no content signal.
MIN_HUNK_CHARS: int = 10


# Hunk parsing
def parse_hunks(patch: str) -> list[str]:
    """Split a raw git diff patch into individual hunks.

    Each hunk begins with a @@ line and includes all subsequent lines
    until the next @@ header or end of string.

    Args:
        patch: Raw patch text from pr_commit_details.patch column.
            May be None or empty.

    Returns:
        List of hunk strings, each starting with its @@ header line.
        Returns an empty list if the patch is None, empty, or contains
        no @@ markers.

    Example:
        >>> hunks = parse_hunks("@@ -1,3 +1,4 @@\\n-old\\n+new\\n")
        >>> len(hunks)
        1
    """
    if not patch or not isinstance(patch, str):
        return []

    # Split on @@ markers; re.split keeps the delimiter via capture group
    parts = re.split(r"(@@ .+? @@[^\n]*\n?)", patch)

    hunks: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.startswith("@@"):
            # Combine the @@ header with the body that follows it
            body = parts[i + 1] if (i + 1) < len(parts) else ""
            hunk = part + body
            if len(hunk.strip()) >= MIN_HUNK_CHARS:
                hunks.append(hunk)
            i += 2
        else:
            i += 1

    return hunks


# PR-level chunk selection
def select_top_k_hunks(
    patches: list[str],
    top_k: int = DEFAULT_TOP_K_HUNKS,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[str]:
    """Extract the first top_k hunks across all patches for a single PR.

    Patches are processed in the order they are provided (file order within
    a commit, commit order within a PR). Selection is first-K, not largest-K,
    to avoid any data-derived ordering that could introduce subtle leakage.

    Args:
        patches: List of raw patch strings for all files in a PR.
            Each element corresponds to one file's diff.
        top_k: Maximum number of hunks to return. Default is 5.
        max_chars: Maximum character length per chunk before passing to the
            tokenizer. Longer hunks are truncated at this boundary.

    Returns:
        List of up to top_k chunk strings, each representing one hunk.
        Returns an empty list if no valid hunks are found across all patches.

    Notes:
        If a PR has fewer than top_k hunks in total, all available hunks
        are returned without padding. The caller (embedding step) should
        handle variable-length chunk lists via mean pooling.
    """
    selected: list[str] = []

    for patch in patches:
        if len(selected) >= top_k:
            break
        for hunk in parse_hunks(patch):
            if len(selected) >= top_k:
                break
            # Truncate at character level before tokenization
            chunk = hunk[:max_chars]
            selected.append(chunk)

    return selected


# DataFrame-level interface
def build_diff_chunks(
    split_df: pd.DataFrame,
    commit_details_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K_HUNKS,
    max_chars: int = MAX_CHARS_PER_CHUNK,
    pr_id_col: str = "id",
) -> pd.DataFrame:
    """Build a chunk list and scalar diff features for each PR in a split.

    This is the primary entry point for Multimodal LR (Model 1) feature extraction.
    It must be called separately for train, val, and test splits to
    ensure no cross-split information is used.

    Args:
        split_df: A single split (train, val, or test) produced by
            splits.py. Must contain a column identified by pr_id_col.
        commit_details_df: The pr_commit_details table filtered to
            the same PRs as split_df. Must contain 'pr_id' and 'patch'
            columns. Also used for scalar diff features if additions/
            deletions/changes columns are present.
        top_k: Number of hunks to select per PR. Default is 5.
        max_chars: Character truncation limit per chunk. Default is 1800.
        pr_id_col: Column name in split_df that holds the PR identifier.
            Default is 'id'.

    Returns:
        DataFrame with one row per PR containing:
            - pr_id: PR identifier
            - label: Binary rejection label (0=accepted, 1=rejected)
            - diff_chunks: List of hunk strings (empty list if no diff)
            - n_chunks: Number of chunks extracted
            - total_changes: Total lines changed (additions + deletions);
                             included as a scalar feature per EDA finding
            - has_diff: Boolean; False when no patch text was found

    Raises:
        KeyError: If pr_id_col is not present in split_df, or if 'pr_id'
            or 'patch' are not present in commit_details_df.

    Example:
        >>> train_features = build_diff_chunks(train_df, commit_details_train)
        >>> train_features.head()
    """
    # Validate required columns
    if pr_id_col not in split_df.columns:
        raise KeyError(
            f"Column '{pr_id_col}' not found in split_df. "
            f"Available columns: {list(split_df.columns)}"
        )
    for col in ("pr_id", "patch"):
        if col not in commit_details_df.columns:
            raise KeyError(
                f"Column '{col}' not found in commit_details_df. "
                f"Available columns: {list(commit_details_df.columns)}"
            )

    logger.info(
        "Building diff chunks for %d PRs (top_k=%d, max_chars=%d)",
        len(split_df),
        top_k,
        max_chars,
    )

    # Compute scalar diff size per PR from commit_details
    scalar_cols = [c for c in ("additions", "deletions", "changes") if c in commit_details_df.columns]
    if scalar_cols:
        scalar_agg = (
            commit_details_df
            .groupby("pr_id")[scalar_cols]
            .sum()
            .reset_index()
        )
        if "additions" in scalar_agg.columns and "deletions" in scalar_agg.columns:
            scalar_agg["total_changes"] = scalar_agg["additions"] + scalar_agg["deletions"]
        elif "changes" in scalar_agg.columns:
            scalar_agg["total_changes"] = scalar_agg["changes"]
        else:
            scalar_agg["total_changes"] = 0
    else:
        logger.warning(
            "No additions/deletions/changes columns found in commit_details_df. "
            "total_changes will be 0 for all PRs."
        )
        scalar_agg = pd.DataFrame(columns=["pr_id", "total_changes"])

    # Group patches by PR, preserving file order within each PR
    patches_by_pr = (
        commit_details_df
        .dropna(subset=["patch"])
        .groupby("pr_id")["patch"]
        .apply(list)
        .reset_index()
        .rename(columns={"patch": "patches"})
    )

    # Join to the split
    result = (
        split_df[[pr_id_col, "label"]]
        .rename(columns={pr_id_col: "pr_id"})
        .merge(patches_by_pr, on="pr_id", how="left")
        .merge(scalar_agg[["pr_id", "total_changes"]], on="pr_id", how="left")
    )

    result["patches"] = result["patches"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    result["total_changes"] = result["total_changes"].fillna(0).astype(int)

    # Apply chunk selection
    result["diff_chunks"] = result["patches"].apply(
        lambda patches: select_top_k_hunks(patches, top_k=top_k, max_chars=max_chars)
    )
    result["n_chunks"] = result["diff_chunks"].apply(len)
    result["has_diff"] = result["n_chunks"] > 0

    # Drop the intermediate patches column
    result = result.drop(columns=["patches"])

    n_empty = (~result["has_diff"]).sum()
    if n_empty > 0:
        logger.warning(
            "%d PRs (%.1f%%) have no extractable diff chunks. "
            "These will produce zero vectors after mean pooling.",
            n_empty,
            100 * n_empty / len(result),
        )

    logger.info(
        "Chunk extraction complete. Median chunks per PR: %.1f. "
        "PRs with no chunks: %d.",
        result["n_chunks"].median(),
        n_empty,
    )

    return result


def drop_empty_chunks(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of diff_df with rows where n_chunks == 0 removed.

    Args:
        diff_df: DataFrame produced by build_diff_chunks. Must contain
            an 'n_chunks' column.

    Returns:
        Filtered DataFrame containing only rows with at least one
        extractable chunk.
    """
    return diff_df[diff_df["n_chunks"] > 0].copy()


def assert_no_empty_chunks(diff_df: pd.DataFrame) -> None:
    """Raise AssertionError if any row has no extractable diff chunks.

    Call this after drop_empty_chunks() as a defensive guard before
    passing data to the embedding step.

    Args:
        diff_df: DataFrame produced by build_diff_chunks (or the
            filtered result of drop_empty_chunks).

    Raises:
        AssertionError: If one or more rows have n_chunks == 0.
    """
    n_empty = int((diff_df["n_chunks"] == 0).sum())
    if n_empty > 0:
        raise AssertionError(
            f"{n_empty} row(s) have no extractable diff chunks. "
            "Run drop_empty_chunks() before passing to the embedding step."
        )
