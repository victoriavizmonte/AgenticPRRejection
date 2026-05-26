"""
Data preprocessing pipeline for AIDev dataset.

This module handles all transformations on top of the raw tables:
- Creating binary labels (merged=0, rejected=1)
- Filtering repos by star count
- Normalizing column names
- Filtering related tables to the main population of PRs
- Merging task type, diff stats, and commit counts
- Adding repo attributes, related issue counts, and per-file aggregates to PRs
- Filtering out Type A (no-commit) and Type B (null-patch) no-diff PRs
- Filtering outlier PRs with excessive total changed lines
- Saving the processed output

Applicability:
  - Applicable to all models. The same preprocessed output feeds all
    model pipelines.
"""

import pandas as pd
from typing import Dict, Optional
import logging

from config import MIN_STARS, PROCESSED_DATA_FILE, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_popular_repos(
    pr_df: pd.DataFrame, repo_df: pd.DataFrame, min_stars: int = MIN_STARS
) -> tuple[pd.DataFrame, set]:
    """
    Keep only PRs that belong to repos with >= min_stars.

    Args:
        pr_df: Raw pull_request DataFrame. May use 'repository_id' or 'repo'
            instead of 'repo_id' — normalized in place.
        repo_df: Raw repository DataFrame. May use 'id' instead of 'repo_id'.
        min_stars: Minimum star count threshold (default MIN_STARS from config).

    Returns:
        Tuple of (filtered pr_df, set of popular repo IDs).
    """
    if "id" in repo_df.columns and "repo_id" not in repo_df.columns:
        repo_df = repo_df.rename(columns={"id": "repo_id"})
    if "repo_id" not in pr_df.columns and "repository_id" in pr_df.columns:
        pr_df = pr_df.rename(columns={"repository_id": "repo_id"})
    if "repo" in pr_df.columns and "repo_id" not in pr_df.columns:
        pr_df = pr_df.rename(columns={"repo": "repo_id"})

    popular_repos = repo_df[repo_df["stars"] >= min_stars]
    popular_repo_ids = set(popular_repos["repo_id"].unique())
    logger.info(f"Popular repos (stars >= {min_stars}): {len(popular_repo_ids)}")

    pr_df = pr_df[pr_df["repo_id"].isin(popular_repo_ids)].copy()
    logger.info(f"PRs in popular repos: {len(pr_df)}")

    return pr_df, popular_repo_ids


def normalize_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Normalize id columns to consistent names (repo_id, pr_id)."""
    df = df.copy()
    if table_name == "pr_task_type":
        if "id" in df.columns and "pr_id" not in df.columns:
            df = df.rename(columns={"id": "pr_id"})
        return df
    if "repository_id" in df.columns and "repo_id" not in df.columns:
        df = df.rename(columns={"repository_id": "repo_id"})
    if "pull_request_id" in df.columns and "pr_id" not in df.columns:
        df = df.rename(columns={"pull_request_id": "pr_id"})
    if "pr" in df.columns and "pr_id" not in df.columns:
        df = df.rename(columns={"pr": "pr_id"})
    return df


def filter_by_repo_or_pr(
    df: pd.DataFrame, repo_ids: set, pr_ids: set
) -> pd.DataFrame:
    """Filter a table to rows matching popular repo IDs or PR IDs.

    Filters by repo_id if that column is present, otherwise by pr_id.
    Returns an empty DataFrame if neither column exists.

    Args:
        df: Table to filter (pr_commits, pr_commit_details, pr_task_type, etc.).
        repo_ids: Set of popular repository IDs to keep.
        pr_ids: Set of PR IDs to keep (used when repo_id column is absent).

    Returns:
        Filtered copy of df containing only matching rows.
    """
    if "repo_id" in df.columns:
        return df[df["repo_id"].isin(repo_ids)].copy()
    if "pr_id" in df.columns:
        return df[df["pr_id"].isin(pr_ids)].copy()
    return df.iloc[0:0].copy()


def create_labels(pr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels and filter to closed PRs only.

    Labels: 0 = merged, 1 = rejected (closed without merge).
    Open PRs are dropped.
    """
    pr_df = pr_df.copy()
    pr_df["is_merged"] = pr_df["merged_at"].notna()
    pr_df["is_rejected"] = (pr_df["state"] == "closed") & pr_df["merged_at"].isna()
    pr_df["is_open"] = pr_df["state"] == "open"

    logger.info(
        f"All PRs — Merged: {pr_df['is_merged'].sum()}, "
        f"Rejected: {pr_df['is_rejected'].sum()}, "
        f"Open: {pr_df['is_open'].sum()}"
    )

    pr_closed = pr_df[~pr_df["is_open"]].copy()
    pr_closed["label"] = pr_closed["is_rejected"].astype(int)

    n_merged = (pr_closed["label"] == 0).sum()
    n_rejected = (pr_closed["label"] == 1).sum()
    logger.info(
        f"Closed PRs: {len(pr_closed)} — "
        f"Merged (0): {n_merged} ({n_merged / len(pr_closed) * 100:.1f}%), "
        f"Rejected (1): {n_rejected} ({n_rejected / len(pr_closed) * 100:.1f}%)"
    )

    return pr_closed

def filter_type_a_nodiff(
    pr_closed: pd.DataFrame,
    pr_commit_details_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove Type A no-diff PRs: PRs absent from pr_commit_details entirely.

    A PR is classified as Type A if it has zero rows in pr_commit_details AND
    num_commits == 0. Using both criteria guards against misclassifying PRs
    that are absent from the details table due to join gaps rather than a
    genuine absence of commits.

    This filter must run after merge_metadata so that num_commits is present.

    Args:
        pr_closed: Closed PR DataFrame with 'id', 'label', and 'num_commits'
            columns. Produced by merge_metadata.
        pr_commit_details_df: The normalized, repo-filtered pr_commit_details
            table used to identify PRs with at least one commit row.

    Returns:
        Filtered DataFrame with Type A PRs removed.
    """
    before = len(pr_closed)
    pr_ids_in_details = set(pr_commit_details_df["pr_id"].unique())

    type_a_mask = (
        (~pr_closed["id"].isin(pr_ids_in_details))
        & (pr_closed["num_commits"] == 0)
    )

    removed = int(type_a_mask.sum())
    filtered = pr_closed[~type_a_mask].copy()

    removed_rej_rate = (
        pr_closed.loc[type_a_mask, "label"].mean() if removed > 0 else float("nan")
    )
    retained_rej_rate = filtered["label"].mean()

    logger.info(f"Type A no-diff filter: {before} PRs before filter")
    logger.info(
        f"Type A no-diff filter: removed {removed} Type A PRs "
        f"(absent from pr_commit_details, num_commits == 0)"
    )
    logger.info(f"Type A no-diff filter: retained {len(filtered)} PRs")
    logger.info(
        f"Type A no-diff filter: rejection rate — "
        f"removed: {removed_rej_rate:.3f}, retained: {retained_rej_rate:.3f}"
    )

    return filtered


def filter_type_b_null_patch_prs(
    pr_closed: pd.DataFrame,
    pr_commit_details_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove Type B no-diff PRs: PRs where every patch value in pr_commit_details
    is null or empty.

    Type B PRs have commit rows in pr_commit_details but all patch values are
    null or empty, producing no extractable diff content. Based on supervisor guidance, 
    These are excluded from the study population entirely rather than being zero-filled.

    This filter runs after filter_type_a_nodiff and before filter_outlier_diffs.

    Args:
        pr_closed: Closed PR DataFrame with an 'id' column.
        pr_commit_details_df: Normalized, repo-filtered pr_commit_details table.

    Returns:
        Filtered DataFrame with Type B null-patch PRs removed.
    """
    before = len(pr_closed)

    has_content = (
        pr_commit_details_df["patch"].notna()
        & (pr_commit_details_df["patch"].str.strip() != "")
    )
    pr_ids_with_content = set(pr_commit_details_df.loc[has_content, "pr_id"].unique())

    filtered = pr_closed[pr_closed["id"].isin(pr_ids_with_content)].copy()
    removed = before - len(filtered)

    logger.info(f"Type B null-patch filter: {before} PRs before filter")
    logger.info(f"Dropped (non-parseable diff): {removed}")
    logger.info(f"Type B null-patch filter: retained {len(filtered)} PRs")

    return filtered


def filter_outlier_diffs(
    pr_closed: pd.DataFrame, max_total_lines: int = 50_000
) -> pd.DataFrame:
    """
    Remove PRs whose total lines changed exceeds max_total_lines.

    Based on EDA. PRs above this threshold are dominated by auto-generated content 
    (e.g. JSON lockfiles).

    Args:
        pr_closed: Closed PR DataFrame with total_changes column.
        max_total_lines: Exclusion threshold (default 50,000).

    Returns:
        Filtered DataFrame with outlier PRs removed.
    """
    before = len(pr_closed)
    mask = pr_closed["total_changes"] <= max_total_lines
    filtered = pr_closed[mask].copy()
    removed = before - len(filtered)

    removed_rej_rate = pr_closed.loc[~mask, "label"].mean() if removed > 0 else float("nan")
    retained_rej_rate = filtered["label"].mean()

    logger.info(
        f"Outlier filter (max_total_lines={max_total_lines:,}): "
        f"removed {removed} PRs ({removed / before * 100:.2f}%), "
        f"retained {len(filtered)} ({len(filtered) / before * 100:.2f}%). "
        f"Rejection rate — removed: {removed_rej_rate:.3f}, retained: {retained_rej_rate:.3f}"
    )

    return filtered

def merge_metadata(
    pr_closed: pd.DataFrame,
    pr_task_type_df: pd.DataFrame,
    pr_commit_details_df: pd.DataFrame,
    pr_commits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge task type, diff stats, and commit counts onto closed PRs.

    Numeric columns from the three left joins (total_additions, total_deletions,
    total_changes, num_files_changed, num_commits) are NaN-filled to 0 for PRs
    with no matching rows. Duplicate pr_id columns introduced by merges are dropped.

    Args:
        pr_closed: Closed PR DataFrame produced by create_labels().
        pr_task_type_df: Normalized pr_task_type table with 'pr_id', 'type',
            'confidence' columns.
        pr_commit_details_df: Normalized, repo-filtered pr_commit_details table
            with 'pr_id', 'additions', 'deletions', 'changes', 'filename'.
        pr_commits_df: Normalized, repo-filtered pr_commits table with 'pr_id'.

    Returns:
        Enriched copy of pr_closed with task_type, task_type_confidence,
        total_additions, total_deletions, total_changes, num_files_changed,
        and num_commits columns added.
    """
    # Task type
    pr_closed = pr_closed.merge(
        pr_task_type_df[["pr_id", "type", "confidence"]].rename(
            columns={"type": "task_type", "confidence": "task_type_confidence"}
        ),
        left_on="id", right_on="pr_id", how="left", suffixes=("", "_task"),
    )

    # Diff stats
    pr_diff_stats = (
        pr_commit_details_df
        .groupby("pr_id")
        .agg(
            total_additions=("additions", "sum"),
            total_deletions=("deletions", "sum"),
            total_changes=("changes", "sum"),
            num_files_changed=("filename", "nunique"),
        )
        .reset_index()
    )
    pr_closed = pr_closed.merge(
        pr_diff_stats, left_on="id", right_on="pr_id",
        how="left", suffixes=("", "_diff"),
    )

    # Commit counts
    pr_commit_counts = (
        pr_commits_df.groupby("pr_id").size().reset_index(name="num_commits")
    )
    pr_closed = pr_closed.merge(
        pr_commit_counts, left_on="id", right_on="pr_id",
        how="left", suffixes=("", "_commits"),
    )

    # Fill NaN numeric columns from left joins
    fill_cols = [
        "total_additions", "total_deletions", "total_changes",
        "num_files_changed", "num_commits",
    ]
    pr_closed[fill_cols] = pr_closed[fill_cols].fillna(0)

    # Drop duplicate pr_id columns created by merges
    pr_id_cols = [c for c in pr_closed.columns if c.startswith("pr_id")]
    pr_closed = pr_closed.drop(columns=pr_id_cols)

    return pr_closed


def _enrich_repo_files_and_issues(
    pr_closed: pd.DataFrame,
    repo_df: pd.DataFrame,
    pr_commit_details_df: pd.DataFrame,
    related_issue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge repository attributes, related issue counts, and per-file aggregates.

    Adds the following columns to pr_closed.parquet:
      - stars, forks, language, license (from repository table)
      - related_issue_count (from related_issue table; 0 if unavailable)
      - filenames (list of filenames per PR from pr_commit_details)
      - file_changes (list of per-file change counts from pr_commit_details)

    Only repository columns that actually exist in repo_df are merged; missing
    columns degrade gracefully (NaN values, imputed later by PRMetadataEncoder).

    Args:
        pr_closed: Closed PR DataFrame with 'id' and 'repo_id' columns.
        repo_df: Repository table. May have 'id' or 'repo_id' as primary key.
        pr_commit_details_df: Filtered commit details with 'pr_id', 'filename', 'changes'.
        related_issue_df: Related issue table with 'pr_id' column, or None.

    Returns:
        Enriched copy of pr_closed.parquet with the columns listed above added.
    """
    pr_closed = pr_closed.copy()

    # 1. Repository attributes
    repo_slim = repo_df.copy()
    if "id" in repo_slim.columns and "repo_id" not in repo_slim.columns:
        repo_slim = repo_slim.rename(columns={"id": "repo_id"})
    repo_attr_cols = [c for c in ["stars", "forks", "language", "license"] if c in repo_slim.columns]
    if repo_attr_cols:
        pr_closed = pr_closed.merge(
            repo_slim[["repo_id"] + repo_attr_cols],
            on="repo_id",
            how="left",
        )

    # 2. related_issue_count
    if related_issue_df is not None and len(related_issue_df) > 0:
        ri_counts = (
            related_issue_df
            .groupby("pr_id")
            .size()
            .reset_index(name="related_issue_count")
        )
        pr_closed = pr_closed.merge(
            ri_counts, left_on="id", right_on="pr_id", how="left"
        )
        pr_closed = pr_closed.drop(columns=["pr_id"], errors="ignore")
        pr_closed["related_issue_count"] = pr_closed["related_issue_count"].fillna(0)
    else:
        pr_closed["related_issue_count"] = 0
    pr_closed["related_issue_count"] = pr_closed["related_issue_count"].astype(float)

    # 3. filenames and file_changes per PR
    file_agg = (
        pr_commit_details_df
        .groupby("pr_id")
        .agg(
            filenames=("filename", list),
            file_changes=("changes", list),
        )
        .reset_index()
    )
    pr_closed = pr_closed.merge(
        file_agg, left_on="id", right_on="pr_id", how="left"
    )
    pr_closed = pr_closed.drop(columns=["pr_id"], errors="ignore")
    pr_closed["filenames"] = pr_closed["filenames"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    pr_closed["file_changes"] = pr_closed["file_changes"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return pr_closed



def run_preprocessing(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Full preprocessing pipeline: filter, normalize, label, merge, and save.

    Pipeline steps:
        1. Filter PRs to repos with >= MIN_STARS stars.
        2. Normalize id column names across all tables.
        3. Filter related tables to the popular-repo/PR subset.
        4. Create binary labels and retain closed PRs only.
        5. Merge task type, diff stats, and commit counts onto closed PRs.
        5b. Merge repository attributes (stars, forks, language, license),
            related issue counts, and per-file aggregates (filenames,
            file_changes) required by PRMetadataEncoder.
        6. Remove Type A no-diff PRs: PRs with zero rows in pr_commit_details
           AND num_commits == 0. These have no commit record at all and their
           100% rejection rate cannot be attributed to code-diff signals.
        7. Remove Type B no-diff PRs: PRs where all patch values in
           pr_commit_details are null or empty. These cannot contribute
           code-diff signals and are excluded entirely.
        8. Remove outlier PRs with total_changes > 50,000 lines.
        9. Save the result to PROCESSED_DATA_FILE.
        10. Log a reminder to manually delete embedding cache files if the PR
            population changed. Caches are position-indexed and must be cleared
            before re-running the embedding extraction cell.

    Args:
        tables: Dict of raw DataFrames as returned by data_loader.load_raw_tables().
            Must include 'pull_request', 'repository', 'pr_commits',
            'pr_commit_details' (required for the no-diff filters at steps 6
            and 7), and 'pr_task_type'. 'related_issue' is optional; if absent,
            related_issue_count is set to 0 for all PRs.

    Returns:
        Processed DataFrame of closed PRs saved to PROCESSED_DATA_FILE.
    """
    pr_df = tables["pull_request"]
    repo_df = tables["repository"]
    pr_commits_df = tables["pr_commits"]
    pr_commit_details_df = tables["pr_commit_details"]
    pr_task_type_df = tables["pr_task_type"]
    related_issue_df = tables.get("related_issue")

    # 1. Filter to popular repos
    pr_df, popular_repo_ids = filter_popular_repos(pr_df, repo_df)
    popular_pr_ids = set(pr_df["id"].unique())

    # 2. Normalize column names
    pr_commits_df = normalize_columns(pr_commits_df, "pr_commits")
    pr_commit_details_df = normalize_columns(pr_commit_details_df, "pr_commit_details")
    pr_task_type_df = normalize_columns(pr_task_type_df, "pr_task_type")

    # 3. Filter related tables
    pr_commits_df = filter_by_repo_or_pr(pr_commits_df, popular_repo_ids, popular_pr_ids)
    pr_commit_details_df = filter_by_repo_or_pr(pr_commit_details_df, popular_repo_ids, popular_pr_ids)
    pr_task_type_df = filter_by_repo_or_pr(pr_task_type_df, popular_repo_ids, popular_pr_ids)
    if related_issue_df is not None:
        related_issue_df = filter_by_repo_or_pr(related_issue_df, popular_repo_ids, popular_pr_ids)

    logger.info(
        f"Filtered: pr_commits={pr_commits_df.shape}, "
        f"pr_commit_details={pr_commit_details_df.shape}, "
        f"pr_task_type={pr_task_type_df.shape}"
    )

    # 4. Create labels & keep closed PRs only
    pr_closed = create_labels(pr_df)

    # 5. Merge metadata
    pr_closed = merge_metadata(pr_closed, pr_task_type_df, pr_commit_details_df, pr_commits_df)

    # 5b. Merge repository attributes, related issue counts, and per-file aggregates
    pr_closed = _enrich_repo_files_and_issues(
        pr_closed, repo_df, pr_commit_details_df, related_issue_df
    )

    # 6. Remove Type A no-diff PRs (absent from pr_commit_details, num_commits == 0)
    pr_closed = filter_type_a_nodiff(pr_closed, pr_commit_details_df)

    # 7. Remove Type B no-diff PRs (commit rows present but all patches null/empty)
    pr_closed = filter_type_b_null_patch_prs(pr_closed, pr_commit_details_df)

    # 8. Filter outlier diffs
    pr_closed = filter_outlier_diffs(pr_closed, max_total_lines=50_000)
    logger.info(f"Final closed PR dataframe: {pr_closed.shape}")

    # 9. Save
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    pr_closed.to_parquet(PROCESSED_DATA_FILE, index=False)
    logger.info(f"Saved {len(pr_closed)} rows to {PROCESSED_DATA_FILE}")

    # Embedding caches are NOT deleted automatically. If the PR population changed,
    # manually delete data/embeddings/model1/*.npy before re-running the embedding cell.
    logger.info("Preprocessing complete. Delete data/embeddings/model1/*.npy manually if the PR set changed.")

    return pr_closed


def ensure_fresh_processed_data() -> None:
    """Check parquet freshness and regenerate if stale or with missing metadata.

    Runs four staleness checks in order: missing file, missing required
    columns, related_issue_count all-zero, and all-empty filenames or
    file_changes.  If any check fails the cached parquet is deleted and
    the full preprocessing pipeline is rerun.

    Uses hasattr/__len__ to test list-like columns so that numpy arrays
    returned by the parquet reader are handled correctly — isinstance(x, list)
    would silently fail for arrays and falsely report every parquet as stale.

    Placing this logic in src/ ensures it is not overwritten when Colab
    auto-saves the notebook back to Drive.
    """
    import pyarrow.parquet as pq

    _REQUIRED_METADATA_COLS = {
        "stars", "forks", "language", "license",
        "filenames", "file_changes", "related_issue_count",
        "base_ref", "title",
    }

    stale_reason: Optional[str] = None

    if not PROCESSED_DATA_FILE.exists():
        stale_reason = "file not found"
    else:
        _cached_cols = set(pq.read_schema(PROCESSED_DATA_FILE).names)
        if not _REQUIRED_METADATA_COLS.issubset(_cached_cols):
            _missing = _REQUIRED_METADATA_COLS - _cached_cols
            stale_reason = f"missing columns: {_missing}"

    if stale_reason is None:
        _rc = pd.read_parquet(PROCESSED_DATA_FILE, columns=["related_issue_count"])
        if _rc["related_issue_count"].sum() == 0:
            stale_reason = "related_issue_count all zeros"

    if stale_reason is None:
        _fc = pd.read_parquet(PROCESSED_DATA_FILE, columns=["filenames", "file_changes"])
        _no_filenames = not _fc["filenames"].apply(
            lambda x: hasattr(x, "__len__") and len(x) > 0
        ).any()
        _no_file_changes = not _fc["file_changes"].apply(
            lambda x: hasattr(x, "__len__") and len(x) > 0
        ).any()
        if _no_filenames or _no_file_changes:
            _cols = (["filenames"] if _no_filenames else []) + (["file_changes"] if _no_file_changes else [])
            stale_reason = f"all-empty: {_cols}"

    if stale_reason is not None:
        logger.info("Stale parquet detected (%s). Regenerating...", stale_reason)
        from data_loader import load_raw_tables
        tables = load_raw_tables([
            "pull_request", "repository", "pr_commits",
            "pr_commit_details", "pr_task_type", "related_issue",
        ])
        run_preprocessing(tables)
        del tables
        logger.info("Preprocessing complete. Parquet saved to %s.", PROCESSED_DATA_FILE)
    else:
        logger.info("Cached parquet is fresh: %s", PROCESSED_DATA_FILE)


def load_processed_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the processed closed-PR parquet file.

    Args:
        sample_size: If provided, return a random sample of this size.

    Returns:
        DataFrame of processed closed PRs, optionally sampled to sample_size rows.

    Raises:
        FileNotFoundError: If processed file doesn't exist yet.
    """
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_FILE}. "
            "Run run_preprocessing() first."
        )

    df = pd.read_parquet(PROCESSED_DATA_FILE)

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
        logger.info(f"Sampled {len(df)} PRs")

    logger.info(f"Loaded {len(df)} closed PRs from {PROCESSED_DATA_FILE}")
    return df
