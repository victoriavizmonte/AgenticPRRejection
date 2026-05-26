"""
Exploratory - Repository-level and agent-level context features for Agentic PR Rejection.

This module encodes per-repository and per-agent submission history into a
fixed-width numeric array (6 features). Features capture historical rejection
tendencies and relative PR size, all derived exclusively from training data.

Applicability:
  - Applicable to Multimodal LR exploration only. Not used in any final
    model pipeline.

Design decisions:
  - Rejection rates : Additive (Laplace) smoothing with the global training
                      rejection rate as prior prevents extreme estimates for
                      low-volume repos or agents.
  - Fallback order  : For the (agent, repo) cross-rate, fall back to the
                      agent-level rate when the cell has < MIN_CELL_COUNT
                      training observations.  For repo- and agent-level rates,
                      fall back to the global training rate for unseen keys.
  - Count features  : log1p to compress right-skew, then StandardScaler fitted
                      on training values.
  - Relative size   : (total_changes - repo_median) / (repo_IQR + 1.0), then
                      StandardScaler fitted on training values.
  - No post-submission fields are used anywhere in this module.
  - Fit on training data only; val/test labels never enter any computation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Module-level constants

SMOOTHING_ALPHA: int = 5
MIN_CELL_COUNT: int = 5

# Sentinel used when agent is null.
_MISSING_AGENT: str = "MISSING"


# Module-level helper function


def _get_repo_id_col(df: pd.DataFrame) -> str:
    """Return the repository identifier column name present in df.

    Checks for 'repo_id' first, then 'repository_id'.  Raises a clear
    KeyError if neither is present.

    Args:
        df: DataFrame to inspect.

    Returns:
        'repo_id' if that column is present in df, otherwise 'repository_id'.

    Raises:
        KeyError: If neither 'repo_id' nor 'repository_id' is in df.columns,
            with a message listing available columns.
    """
    if "repo_id" in df.columns:
        return "repo_id"
    if "repository_id" in df.columns:
        return "repository_id"
    raise KeyError(
        "Neither 'repo_id' nor 'repository_id' found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


# RepositoryContextEncoder


class RepositoryContextEncoder:
    """Sklearn-compatible encoder for repository- and agent-level context features.

    Computes six features derived exclusively from training-split statistics.
    At transform time, unseen repositories or agents fall back to
    global or agent-level rates.

    Feature list (output column order):
      1. repo_rejection_rate        — smoothed rejection rate for the PR's repo
      2. agent_rejection_rate       — smoothed rejection rate for the PR's agent
      3. agent_repo_rejection_rate  — smoothed rate for the (agent, repo) cross-
                                      cell; falls back to agent_rejection_rate
                                      when the cell has < MIN_CELL_COUNT PRs
      4. log1p_repo_pr_count        — StandardScaler(log1p(# training PRs in repo))
      5. log1p_agent_pr_count       — StandardScaler(log1p(# training PRs by agent))
      6. pr_size_vs_repo_median     — StandardScaler((total_changes - repo_median)
                                       / (repo_IQR + 1.0))

    Rejection rate columns are not scaled — they are bounded in [0, 1] by
    construction.  Count and relative-size columns are StandardScaler-normalized
    using parameters fitted on the training split only.

    Attributes:
        feature_names_: Ordered list of output column names, set after fit().

    Example:
        >>> enc = RepositoryContextEncoder()
        >>> X_ctx_train = enc.fit_transform(train_df)
        >>> X_ctx_val   = enc.transform(val_df)
        >>> assert X_ctx_train.shape[1] == 6
        >>> assert enc.feature_names_[0] == "repo_rejection_rate"
    """

    def __init__(self) -> None:
        """Initialise the encoder without fitting any statistics."""
        self._global_rejection_rate: float = 0.0
        self._global_median_changes: float = 0.0
        self._global_iqr_changes: float = 0.0

        # Per-repo statistics
        self._repo_smoothed_rates: dict[str, float] = {}
        self._repo_pr_counts: dict[str, int] = {}
        self._repo_median_changes: dict[str, float] = {}
        self._repo_iqr_changes: dict[str, float] = {}

        # Per-agent statistics
        self._agent_smoothed_rates: dict[str, float] = {}
        self._agent_pr_counts: dict[str, int] = {}

        # Per-(agent, repo) statistics
        self._agent_repo_smoothed_rates: dict[tuple, float] = {}
        self._agent_repo_counts: dict[tuple, int] = {}

        self._count_scaler: Optional[StandardScaler] = None
        self._size_scaler: Optional[StandardScaler] = None

        self.feature_names_: Optional[list[str]] = None
        self._fitted: bool = False

    # Public interface
    def fit(self, df: pd.DataFrame) -> "RepositoryContextEncoder":
        """Fit all statistics and scalers on the training DataFrame.

        Computes per-repository and per-agent rejection rates with additive
        smoothing, per-repository size statistics (median and IQR of
        total_changes), and fits StandardScaler instances for count and
        relative-size features.  All statistics derive exclusively from df.
        Never call this on val or test splits.

        Args:
            df: Training DataFrame.  Must contain the repository identifier
                column ('repo_id' or 'repository_id'), 'agent',
                'total_changes', and 'label'.

        Returns:
            self, to allow method chaining.

        Raises:
            KeyError: If any required column is absent from df.
            ValueError: If df is empty.
        """
        _get_repo_id_col(df)  # validates repo col; raises KeyError if absent
        self._validate_required_columns(df, include_label=True)

        if len(df) == 0:
            raise ValueError(
                "Cannot fit RepositoryContextEncoder on an empty DataFrame."
            )

        repo_col = _get_repo_id_col(df)

        # Global statistics
        self._global_rejection_rate = float(df["label"].mean())
        changes_all = df["total_changes"].fillna(0.0).astype(float).values
        self._global_median_changes = float(np.median(changes_all))
        q75_all = float(np.percentile(changes_all, 75))
        q25_all = float(np.percentile(changes_all, 25))
        self._global_iqr_changes = q75_all - q25_all

        self._fit_repo_stats(df, repo_col)
        self._fit_agent_stats(df)
        self._fit_agent_repo_stats(df, repo_col)
        self._fit_scalers(df, repo_col)

        self.feature_names_ = [
            "repo_rejection_rate",
            "agent_rejection_rate",
            "agent_repo_rejection_rate",
            "log1p_repo_pr_count",
            "log1p_agent_pr_count",
            "pr_size_vs_repo_median",
        ]
        self._fitted = True
        logger.info(
            "RepositoryContextEncoder fitted on %d PRs. "
            "Global rejection rate: %.4f. "
            "Repos: %d. Agents: %d. (Agent, repo) cells: %d.",
            len(df),
            self._global_rejection_rate,
            len(self._repo_smoothed_rates),
            len(self._agent_smoothed_rates),
            len(self._agent_repo_counts),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into the 6-feature context array.

        Looks up per-repo and per-agent statistics fitted during fit().
        Unseen repository IDs or agents fall back to the global training
        rejection rate.  The (agent, repo) cross-rate falls back to the
        agent-level rate when the cell had < MIN_CELL_COUNT training PRs.

        Args:
            df: DataFrame containing the same columns as fit()'s input.
                May contain repository IDs or agent names not seen in training.
                The 'label' column is not required at transform time.

        Returns:
            np.ndarray of shape (n_samples, 6) with dtype float32.
            Column order matches feature_names_.

        Raises:
            RuntimeError: If fit() has not been called first.
            KeyError: If any required column is absent from df.
        """
        if not self._fitted:
            raise RuntimeError(
                "RepositoryContextEncoder.fit() must be called before transform(). "
                "Use fit_transform() to do both in one step."
            )
        _get_repo_id_col(df)
        self._validate_required_columns(df, include_label=False)

        repo_col = _get_repo_id_col(df)
        agent_keys = df["agent"].fillna(_MISSING_AGENT).astype(str)

        # Feature 1: repo_rejection_rate
        repo_rates = df[repo_col].apply(
            lambda r: self._repo_smoothed_rates.get(r, self._global_rejection_rate)
        ).values.astype(float)

        # Feature 2: agent_rejection_rate
        agent_rates = agent_keys.apply(
            lambda a: self._agent_smoothed_rates.get(a, self._global_rejection_rate)
        ).values.astype(float)

        # Feature 3: agent_repo_rejection_rate (with per-cell fallback)
        agent_repo_rates = self._compute_agent_repo_rates(
            df, repo_col, agent_keys
        )

        # Feature 4 & 5: log1p count features (scaled)
        log_repo_counts = np.log1p(
            df[repo_col].map(self._repo_pr_counts).fillna(0).values.astype(float)
        ).reshape(-1, 1)
        log_agent_counts = np.log1p(
            agent_keys.map(self._agent_pr_counts).fillna(0).values.astype(float)
        ).reshape(-1, 1)
        count_matrix = np.hstack([log_repo_counts, log_agent_counts])
        scaled_counts = self._count_scaler.transform(count_matrix)

        # Feature 6: pr_size_vs_repo_median (scaled)
        repo_medians = df[repo_col].apply(
            lambda r: self._repo_median_changes.get(r, self._global_median_changes)
        ).values.astype(float)
        repo_iqrs = df[repo_col].apply(
            lambda r: self._repo_iqr_changes.get(r, self._global_iqr_changes)
        ).values.astype(float)
        changes = df["total_changes"].fillna(0.0).astype(float).values
        relative_size = ((changes - repo_medians) / (repo_iqrs + 1.0)).reshape(-1, 1)
        scaled_size = self._size_scaler.transform(relative_size)

        result = np.column_stack([
            repo_rates,
            agent_rates,
            agent_repo_rates,
            scaled_counts[:, 0],
            scaled_counts[:, 1],
            scaled_size[:, 0],
        ])
        return result.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df, then transform df.

        Equivalent to fit(df).transform(df).  Provided for sklearn
        compatibility and convenience.

        Args:
            df: Training DataFrame containing all required columns.

        Returns:
            np.ndarray of shape (n_samples, 6) with dtype float32.
        """
        return self.fit(df).transform(df)


    # Private fit helpers
    def _fit_repo_stats(self, df: pd.DataFrame, repo_col: str) -> None:
        """Compute and store per-repository statistics from training data.

        For each repository, stores the smoothed rejection rate, total PR
        count, and the median and IQR of total_changes.

        Args:
            df: Training DataFrame.
            repo_col: Column name for the repository identifier.
        """
        global_rate = self._global_rejection_rate
        for repo_id, group in df.groupby(repo_col):
            n_total = len(group)
            n_rejected = int(group["label"].sum())
            smoothed = (n_rejected + SMOOTHING_ALPHA * global_rate) / (
                n_total + SMOOTHING_ALPHA
            )
            changes = group["total_changes"].fillna(0.0).astype(float).values
            q75 = float(np.percentile(changes, 75))
            q25 = float(np.percentile(changes, 25))
            self._repo_smoothed_rates[repo_id] = smoothed
            self._repo_pr_counts[repo_id] = n_total
            self._repo_median_changes[repo_id] = float(np.median(changes))
            self._repo_iqr_changes[repo_id] = q75 - q25

        logger.info("Fitted repo stats for %d repositories.", len(self._repo_smoothed_rates))

    def _fit_agent_stats(self, df: pd.DataFrame) -> None:
        """Compute and store per-agent statistics from training data.

        For each agent, stores the smoothed rejection rate and total PR count.
        Null agent values are treated as _MISSING_AGENT.

        Args:
            df: Training DataFrame.
        """
        global_rate = self._global_rejection_rate
        tmp = df.copy()
        tmp["_agent_key"] = df["agent"].fillna(_MISSING_AGENT).astype(str)
        for agent, group in tmp.groupby("_agent_key"):
            n_total = len(group)
            n_rejected = int(group["label"].sum())
            smoothed = (n_rejected + SMOOTHING_ALPHA * global_rate) / (
                n_total + SMOOTHING_ALPHA
            )
            self._agent_smoothed_rates[agent] = smoothed
            self._agent_pr_counts[agent] = n_total

        logger.info("Fitted agent stats for %d agents.", len(self._agent_smoothed_rates))

    def _fit_agent_repo_stats(self, df: pd.DataFrame, repo_col: str) -> None:
        """Compute and store per-(agent, repo) cell statistics from training data.

        Cells with < MIN_CELL_COUNT observations are retained in the lookup
        but the count is stored so transform() can decide whether to fall
        back to the agent-level rate.

        Args:
            df: Training DataFrame.
            repo_col: Column name for the repository identifier.
        """
        global_rate = self._global_rejection_rate
        tmp = df.copy()
        tmp["_agent_key"] = df["agent"].fillna(_MISSING_AGENT).astype(str)
        for (agent, repo_id), group in tmp.groupby(["_agent_key", repo_col]):
            n_total = len(group)
            n_rejected = int(group["label"].sum())
            smoothed = (n_rejected + SMOOTHING_ALPHA * global_rate) / (
                n_total + SMOOTHING_ALPHA
            )
            key = (agent, repo_id)
            self._agent_repo_smoothed_rates[key] = smoothed
            self._agent_repo_counts[key] = n_total

        n_sparse = sum(
            1 for cnt in self._agent_repo_counts.values() if cnt < MIN_CELL_COUNT
        )
        logger.info(
            "Fitted (agent, repo) stats for %d cells "
            "(%d with < %d obs, will fall back to agent rate at transform time).",
            len(self._agent_repo_counts),
            n_sparse,
            MIN_CELL_COUNT,
        )

    def _fit_scalers(self, df: pd.DataFrame, repo_col: str) -> None:
        """Fit StandardScaler instances for count and relative-size features.

        Both scalers are fitted on training-set values only.  Count features
        are log1p-transformed before fitting.

        Args:
            df: Training DataFrame.
            repo_col: Column name for the repository identifier.
        """
        log_repo = np.log1p(
            df[repo_col].map(self._repo_pr_counts).fillna(0).values.astype(float)
        ).reshape(-1, 1)
        log_agent = np.log1p(
            df["agent"].fillna(_MISSING_AGENT).astype(str).map(
                self._agent_pr_counts
            ).fillna(0).values.astype(float)
        ).reshape(-1, 1)
        self._count_scaler = StandardScaler()
        self._count_scaler.fit(np.hstack([log_repo, log_agent]))

        repo_medians = df[repo_col].map(self._repo_median_changes).fillna(
            self._global_median_changes
        ).values.astype(float)
        repo_iqrs = df[repo_col].map(self._repo_iqr_changes).fillna(
            self._global_iqr_changes
        ).values.astype(float)
        changes = df["total_changes"].fillna(0.0).astype(float).values
        relative_size = ((changes - repo_medians) / (repo_iqrs + 1.0)).reshape(-1, 1)
        self._size_scaler = StandardScaler()
        self._size_scaler.fit(relative_size)

    # Private transform helpers
    def _compute_agent_repo_rates(
        self,
        df: pd.DataFrame,
        repo_col: str,
        agent_keys: pd.Series,
    ) -> np.ndarray:
        """Compute per-row agent_repo_rejection_rate with cell-count fallback.

        Returns the smoothed (agent, repo) cross-rate when the cell had
        >= MIN_CELL_COUNT training observations, otherwise returns the
        agent-level rate (or global rate if the agent is also unseen).

        Args:
            df: DataFrame being transformed.
            repo_col: Column name for the repository identifier.
            agent_keys: Series of normalized agent strings (NaN -> _MISSING_AGENT),
                aligned with df's index.

        Returns:
            np.ndarray of shape (n_samples,) with dtype float64.
        """
        rates = np.empty(len(df), dtype=float)
        repo_vals = df[repo_col].values
        agent_vals = agent_keys.values
        for i in range(len(df)):
            agent = agent_vals[i]
            repo = repo_vals[i]
            key = (agent, repo)
            if (
                key in self._agent_repo_counts
                and self._agent_repo_counts[key] >= MIN_CELL_COUNT
            ):
                rates[i] = self._agent_repo_smoothed_rates[key]
            else:
                rates[i] = self._agent_smoothed_rates.get(
                    agent, self._global_rejection_rate
                )
        return rates

    # Private utilities
    def _validate_required_columns(
        self, df: pd.DataFrame, include_label: bool
    ) -> None:
        """Raise KeyError if any required column is absent from df.

        The repository identifier column is checked separately via
        _get_repo_id_col() before this method is called.

        Args:
            df: DataFrame to validate.
            include_label: If True, also check for the 'label' column
                (required during fit; not required during transform).

        Raises:
            KeyError: Lists all missing columns in the error message.
        """
        required = ["agent", "total_changes"]
        if include_label:
            required.append("label")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"RepositoryContextEncoder is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
