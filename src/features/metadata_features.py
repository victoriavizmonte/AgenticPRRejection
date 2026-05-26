"""
Metadata feature extraction for Agentic PR Rejection prediction.

This module encodes structured, submission-time PR metadata into a fixed-width
numeric feature vector. Covers repository signals, code-change statistics, agent
and task categoricals, file-type flags, plan-detection, submission boolean signals,
cyclic time encodings, and a Shannon entropy measure of diff spread.

Applicability:
  - Module-level helper functions (_compute_entropy, _detect_has_plan,
    _empty_description, _is_experimental_submission, _is_test_related,
    _touches_ci, _touches_config, _touches_tests): applicable to all models.
    Standard LLM and reasoning LLM classifiers import these directly to compute
    derived signals that are formatted into the LLM prompt text.
  - PRMetadataEncoder class: applicable to Multimodal LR only. Produces the
    full fixed-width numeric feature vector for logistic regression. LLM
    classifiers do not use this class.

Design decisions:
  - Numeric: log1p transformation before StandardScaler to handle right-skew.
    Medians computed on non-null train values; NaN imputed with median.
  - Categorical: OneHotEncoder with an explicit UNKNOWN category bucket.
    Unseen categories at transform time map to UNKNOWN (not zeroed).
  - Flags: Derived from a list-valued `filenames` column (one list per PR).
  - Entropy: Derived from a list-valued `file_changes` column.
  - Interface: sklearn-style fit / transform / fit_transform for compatibility
    with permutation importance and pipeline composition.
  - No post-submission fields are used anywhere in this module.
"""

import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Public column-name constants

NUMERIC_COLS: list[str] = [
    "stars",
    "forks",
    "total_additions",
    "total_deletions",
    "total_changes",
    "num_files_changed",
    "num_commits",
    "related_issue_count",
    # body_length is derived from 'body' inside the module
]

CATEGORICAL_COLS: list[str] = ["agent", "language", "license", "task_type"]

# Module-level constants for file-type pattern matching
_PLAN_PATTERN: re.Pattern = re.compile(r"(?i)(plan:|steps:|approach:|todo:)")

_CI_EXTENSIONS: frozenset[str] = frozenset([".yml", ".yaml", ".sh"])
_CI_SUBSTRINGS: list[str] = [".github/", "Dockerfile"]

_TEST_SUBSTRINGS: list[str] = ["test_", "_test.", "/tests/", "/test/", ".test.", "spec."]

_CONFIG_EXTENSIONS: frozenset[str] = frozenset([".json", ".toml", ".ini", ".cfg"])
_CONFIG_SUBSTRINGS: list[str] = ["setup.py", "requirements.txt"]

_EXPERIMENTAL_KEYWORDS: list[str] = [
    "experiment",
    "trying out",
    "test run",
    "see if",
    "just testing",
    "evaluate",
    "proof of concept",
    "poc",
]

# Used for categorical imputation when the value is null.
_MISSING: str = "MISSING"
# Bucket for categories not seen during fit.
_UNKNOWN: str = "UNKNOWN"


# Module-level helper functions
def _touches_ci(filenames: list[str]) -> bool:
    """Return True if any filename in the PR matches a CI-related pattern.

    Matches on file extension (.yml, .yaml, .sh) or substring (.github/,
    Dockerfile). The check is case-sensitive because filesystem paths are
    case-sensitive on Linux (the target Colab environment).

    Args:
        filenames: List of filenames changed in the PR. May be empty.

    Returns:
        True if at least one filename matches a CI pattern, False otherwise.
    """
    for f in filenames:
        ext = os.path.splitext(f)[1]
        if ext in _CI_EXTENSIONS:
            return True
        for sub in _CI_SUBSTRINGS:
            if sub in f:
                return True
    return False


def _touches_tests(filenames: list[str]) -> bool:
    """Return True if any filename in the PR matches a test-file pattern.

    Matches substrings: test_, _test., /tests/, spec.

    Args:
        filenames: List of filenames changed in the PR.  May be empty.

    Returns:
        True if at least one filename matches a test pattern, False otherwise.
    """
    for f in filenames:
        for sub in _TEST_SUBSTRINGS:
            if sub in f:
                return True
    return False


def _touches_config(filenames: list[str]) -> bool:
    """Return True if any filename in the PR matches a config-file pattern.

    Matches on extension (.json, .toml, .ini, .cfg) or exact filename suffix
    (setup.py, requirements.txt).

    Args:
        filenames: List of filenames changed in the PR. May be empty.

    Returns:
        True if at least one filename matches a config pattern, False otherwise.
    """
    for f in filenames:
        ext = os.path.splitext(f)[1]
        if ext in _CONFIG_EXTENSIONS:
            return True
        for sub in _CONFIG_SUBSTRINGS:
            if sub in f:
                return True
    return False


def _compute_entropy(changes: list[float]) -> float:
    """Compute the Shannon entropy of a per-file change distribution.

    A PR whose changes are concentrated in one file has entropy 0.  A PR
    whose changes are spread evenly across many files has higher entropy.
    Uses natural logarithm.

    Args:
        changes: List of numeric change counts (one per file). Values of
        zero or below are excluded from the distribution. An empty list
        or a list of a single positive value yields 0.0.

    Returns:
        Shannon entropy in nats, as a non-negative float. Returns 0.0 for
        empty, all-zero, or single-element distributions.
    """
    positive = [c for c in changes if c > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probs = np.array(positive, dtype=float) / total
    return float(-np.sum(probs * np.log(probs)))


def _detect_has_plan(body: object) -> bool:
    """Return True if the PR body contains a planning keyword.

    Searches for plan:, steps:, approach:, or todo: (case-insensitive).
    A null or NaN body is treated as an empty string and returns False.

    Args:
        body: Raw PR body text. May be None, NaN, or any string type.

    Returns:
        True if a planning keyword is found, False otherwise.
    """
    if body is None or (isinstance(body, float) and np.isnan(body)):
        return False
    return bool(_PLAN_PATTERN.search(str(body)))


def _is_experimental_submission(title: object, body: object) -> bool:
    """Return True if the PR title or body signals an experimental intent.

    Checks for the presence of any keyword from _EXPERIMENTAL_KEYWORDS in the
    combined title and body text (case-insensitive).  Null or NaN values are
    treated as empty strings.

    Args:
        title: PR title text. May be None, NaN, or str.
        body: PR body text. May be None, NaN, or str.

    Returns:
        True if any experimental keyword is found, False otherwise.
    """
    def _to_str(val: object) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return str(val)

    combined = (_to_str(title) + " " + _to_str(body)).lower()
    return any(kw in combined for kw in _EXPERIMENTAL_KEYWORDS)


def _is_test_related(title: object, body: object, task_type: object) -> bool:
    """Return True if the PR title, body, or task_type signals test-related work.

    Checks two conditions:
      1. task_type equals "test" (case-insensitive).
      2. The PR title or body contains the word "test" at a word boundary
         (case-insensitive), matching "test", "tests", "testing", "tested",
         "unittest", "pytest", etc.

    This captures test-related PRs that do not touch a test file in the diff
    (e.g., the agent describes adding tests in the body without creating new
    test files, or the dataset labels the PR as task_type="test").

    Args:
        title: PR title text. May be None, NaN, or str.
        body: PR body text. May be None, NaN, or str.
        task_type: Task type label string. May be None, NaN, or str.

    Returns:
        True if the PR is test-related by task_type or text content.
    """
    def _to_str(val: object) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return str(val)

    if _to_str(task_type).lower() == "test":
        return True
    combined = (_to_str(title) + " " + _to_str(body)).lower()
    return bool(re.search(r"\btest", combined))


def _empty_description(body: object) -> bool:
    """Return True if the PR body is null, empty, or whitespace-only.

    Args:
        body: PR body text.  May be None, NaN, or str.

    Returns:
        True if body carries no textual content, False otherwise.
    """
    if body is None or (isinstance(body, float) and np.isnan(body)):
        return True
    return str(body).strip() == ""


# PRMetadataEncoder


class PRMetadataEncoder:
    """Sklearn-compatible encoder for structured PR metadata features.

    Applicable to Multimodal LR only. Standard LLM and reasoning LLM classifiers
    import the module-level helper functions directly rather than this class.

    Transforms a PR-level DataFrame into a fixed-width numeric array suitable
    for logistic regression, gradient-boosted models, or use as a feature
    block alongside text embeddings and diff representations.

    The encoder covers seven feature groups:
      1. Numeric (log1p + StandardScaler): stars, forks, additions, deletions,
         changes, files changed, commits, related issues, body length.
      2. Categorical (OneHotEncoder with UNKNOWN bucket): agent, language,
         license, task_type.
      3. Binary flags: touches_ci, touches_tests, touches_config.
      4. Boolean: has_plan (detected from body text via regex).
      5. Submission booleans: is_experimental_submission, empty_description.
      6. Cyclic time: sin/cos encoding of hour-of-day and day-of-week.
      7. Entropy: Shannon entropy of the per-file change distribution.

    All transformers are fitted on training data only. Call fit() (or
    fit_transform()) on the training split, then transform() on val and test.

    Attributes:
        feature_names_: Ordered list of output column names, set after fit().

    Example:
        >>> enc = PRMetadataEncoder()
        >>> X_train = enc.fit_transform(train_df)
        >>> X_val   = enc.transform(val_df)
        >>> assert X_train.shape[1] == len(enc.feature_names_)
    """

    def __init__(self, min_category_freq: int = 20) -> None:
        """Initialise the encoder without fitting any transformers.

        Args:
            min_category_freq: Minimum number of training-set occurrences a
                category must have to be kept as its own OHE column.  Any
                category with fewer occurrences is collapsed into the UNKNOWN
                bucket before the OHE is fitted, preventing sparse categories
                from introducing noise.  MISSING is never collapsed regardless
                of its frequency.  Default is 20.
        """
        self._min_category_freq = min_category_freq
        self._scaler: Optional[StandardScaler] = None
        self._ohe: Optional[OneHotEncoder] = None
        self._cat_known_values: dict[str, set[str]] = {}
        self._rare_categories: dict[str, set[str]] = {}
        self._numeric_medians: dict[str, float] = {}
        self.feature_names_: Optional[list[str]] = None
        self._fitted: bool = False

    # Public interface
    def fit(self, df: pd.DataFrame) -> "PRMetadataEncoder":
        """Fit all transformers on the training DataFrame.

        Computes and stores: numeric medians, StandardScaler parameters, and
        OneHotEncoder category lists (including MISSING and UNKNOWN buckets).
        Sets feature_names_ after fitting.

        Args:
            df: Training DataFrame containing all required columns (see module
                docstring for the full input contract).

        Returns:
            self, to allow method chaining.

        Raises:
            KeyError: If any required column is absent from df.
            ValueError: If df is empty.
        """
        self._validate_columns(df)
        if len(df) == 0:
            raise ValueError("Cannot fit PRMetadataEncoder on an empty DataFrame.")

        body_length = self._compute_body_length(df)
        self._fit_numerics(df, body_length)
        self._fit_categoricals(df)

        self.feature_names_ = self._build_feature_names()
        self._fitted = True
        logger.info(
            "PRMetadataEncoder fitted. Total features: %d.", len(self.feature_names_)
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into a numeric feature matrix.

        All transformers are applied using parameters fitted during fit().
        Unseen categorical values are mapped to the UNKNOWN bucket.  Null
        values in every field are handled as documented in the module header.

        Args:
            df: DataFrame containing all required columns.  May contain nulls.

        Returns:
            np.ndarray of shape (n_samples, n_features) with dtype float64.
            Column order matches feature_names_.

        Raises:
            RuntimeError: If fit() has not been called first.
            KeyError: If any required column is absent from df.
        """
        if not self._fitted:
            raise RuntimeError(
                "PRMetadataEncoder.fit() must be called before transform(). "
                "Use fit_transform() to do both in one step."
            )
        self._validate_columns(df)

        parts: list[np.ndarray] = [
            self._transform_numerics(df),
            self._transform_categoricals(df),
            self._transform_flags(df),
            self._transform_has_plan(df),
            self._transform_submission_booleans(df),
            self._transform_cyclic(df),
            self._transform_entropy(df),
        ]
        return np.hstack(parts).astype(np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df, then transform df.

        Equivalent to fit(df).transform(df).  Provided for sklearn
        compatibility and convenience.

        Args:
            df: Training DataFrame containing all required columns.

        Returns:
            np.ndarray of shape (n_samples, n_features) with dtype float64.
        """
        return self.fit(df).transform(df)

    # Private fit helpers
    def _fit_numerics(self, df: pd.DataFrame, body_length: pd.Series) -> None:
        """Compute medians and fit StandardScaler on log1p-transformed values.

        Medians are computed on non-null values only (pandas .median() ignores
        NaN by default).  NaN values are then filled with the stored median
        before log1p and scaling.

        Args:
            df: Training DataFrame.
            body_length: Pre-computed body length Series (index matches df).
        """
        numeric_data: dict[str, pd.Series] = {}
        for col in NUMERIC_COLS:
            series = df[col].copy()
            median_val = float(series.median())
            self._numeric_medians[col] = median_val
            numeric_data[col] = series.fillna(median_val)

        # body_length has no NaN (computed from fillna(''))
        self._numeric_medians["body_length"] = float(body_length.median())
        numeric_data["body_length"] = body_length

        matrix = np.column_stack([numeric_data[c].values for c in self._all_numeric_cols()])
        matrix = np.log1p(matrix.astype(float))

        self._scaler = StandardScaler()
        self._scaler.fit(matrix)

    def _fit_categoricals(self, df: pd.DataFrame) -> None:
        """Collect category lists and fit OneHotEncoder.

        For each categorical column, categories with fewer than
        min_category_freq occurrences in training are collapsed into the
        UNKNOWN bucket before the OHE is fitted.  MISSING is never
        collapsed.  The remaining known categories plus UNKNOWN form the
        OHE vocabulary.  At transform time, any value not in the known set
        is mapped to UNKNOWN before encoding.

        Args:
            df: Training DataFrame.
        """
        categories_per_col: list[list[str]] = []
        for col in CATEGORICAL_COLS:
            vals = df[col].fillna(_MISSING).astype(str)
            counts = vals.value_counts()
            rare = {
                cat for cat, cnt in counts.items()
                if cnt < self._min_category_freq and cat not in (_MISSING, _UNKNOWN)
            }
            self._rare_categories[col] = rare
            if rare:
                vals = vals.where(~vals.isin(rare), other=_UNKNOWN)
            known = sorted(vals.unique().tolist())
            if _UNKNOWN not in known:
                known.append(_UNKNOWN)
            self._cat_known_values[col] = set(known)
            categories_per_col.append(known)

        cat_matrix = self._build_cat_matrix(df)
        self._ohe = OneHotEncoder(
            categories=categories_per_col,
            handle_unknown="ignore",
            sparse_output=False,
        )
        self._ohe.fit(cat_matrix)

    # Private transform helpers
    def _transform_numerics(self, df: pd.DataFrame) -> np.ndarray:
        """Apply stored medians, log1p, and StandardScaler.

        Args:
            df: DataFrame to transform.

        Returns:
            np.ndarray of shape (n_samples, 9).
        """
        body_length = self._compute_body_length(df)
        cols: list[np.ndarray] = []
        for col in NUMERIC_COLS:
            series = df[col].fillna(self._numeric_medians[col])
            cols.append(series.values)
        cols.append(body_length.values)

        matrix = np.column_stack(cols).astype(float)
        matrix = np.log1p(matrix)
        return self._scaler.transform(matrix)

    def _transform_categoricals(self, df: pd.DataFrame) -> np.ndarray:
        """Replace unseen categories with UNKNOWN, then one-hot encode.

        Args:
            df: DataFrame to transform.

        Returns:
            np.ndarray of shape (n_samples, n_ohe_features).
        """
        cat_matrix = self._build_cat_matrix(df)
        return self._ohe.transform(cat_matrix)

    def _transform_flags(self, df: pd.DataFrame) -> np.ndarray:
        """Compute binary file-type flags from filenames and PR text fields.

        The touches_ci and touches_config flags derive from the 'filenames'
        column only.  The touches_tests flag is broader: it OR-s filename
        pattern matching (_touches_tests) with text-based detection
        (_is_test_related), which also reads 'title', 'body', and 'task_type'.

        Args:
            df: DataFrame with 'filenames' (list of strings per row), 'title',
                'body', and 'task_type' columns.

        Returns:
            np.ndarray of shape (n_samples, 3): [touches_ci, touches_tests,
            touches_config].
        """
        filenames_col = df["filenames"].apply(
            lambda x: [f for f in x if isinstance(f, str)] if isinstance(x, (list, np.ndarray)) else []
        )
        ci = filenames_col.apply(_touches_ci).astype(float).values
        tests = df.apply(
            lambda row: (
                _touches_tests([f for f in row["filenames"] if isinstance(f, str)] if isinstance(row.get("filenames"), (list, np.ndarray)) else [])
                or _is_test_related(row.get("title"), row.get("body"), row.get("task_type"))
            ),
            axis=1,
        ).astype(float).values
        config = filenames_col.apply(_touches_config).astype(float).values
        return np.column_stack([ci, tests, config])

    def _transform_has_plan(self, df: pd.DataFrame) -> np.ndarray:
        """Detect planning keywords in PR body text.

        Args:
            df: DataFrame with a 'body' column.

        Returns:
            np.ndarray of shape (n_samples, 1).
        """
        has_plan = df["body"].apply(_detect_has_plan).astype(float).values
        return has_plan.reshape(-1, 1)

    def _transform_submission_booleans(self, df: pd.DataFrame) -> np.ndarray:
        """Compute submission-level boolean signals from title and body.

        Args:
            df: DataFrame with 'title' and 'body' columns.

        Returns:
            np.ndarray of shape (n_samples, 2): [is_experimental_submission,
            empty_description].
        """
        experimental = (
            df.apply(
                lambda row: _is_experimental_submission(row["title"], row["body"]),
                axis=1,
            )
            .astype(float)
            .values
        )
        empty_desc = (
            df["body"].apply(_empty_description).astype(float).values
        )
        return np.column_stack([experimental, empty_desc])

    def _transform_cyclic(self, df: pd.DataFrame) -> np.ndarray:
        """Encode submission hour and day-of-week as sin/cos pairs.

        Args:
            df: DataFrame with a 'created_at' column (datetime or str).

        Returns:
            np.ndarray of shape (n_samples, 4): [hour_sin, hour_cos,
            dow_sin, dow_cos].

        Raises:
            ValueError: If any value in 'created_at' is null.
        """
        ts = pd.to_datetime(df["created_at"])
        if ts.isna().any():
            raise ValueError(
                "'created_at' contains null values. All PRs must have a "
                "creation timestamp for cyclic time encoding."
            )
        hour = ts.dt.hour.astype(float)
        dow = ts.dt.dayofweek.astype(float)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        return np.column_stack([hour_sin, hour_cos, dow_sin, dow_cos])

    def _transform_entropy(self, df: pd.DataFrame) -> np.ndarray:
        """Compute per-PR Shannon entropy of the file change distribution.

        Args:
            df: DataFrame with a 'file_changes' column (list of numeric
                values per row).

        Returns:
            np.ndarray of shape (n_samples, 1).
        """
        changes_col = df["file_changes"].apply(
            lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
        )
        entropy_vals = changes_col.apply(_compute_entropy).astype(float).values
        return entropy_vals.reshape(-1, 1)

    # Private utilities
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise KeyError if any required input column is absent.

        Args:
            df: DataFrame to validate.

        Raises:
            KeyError: Lists all missing columns in the error message.
        """
        required = (
            NUMERIC_COLS
            + CATEGORICAL_COLS
            + ["body", "filenames", "file_changes", "created_at", "title"]
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"PRMetadataEncoder is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    @staticmethod
    def _compute_body_length(df: pd.DataFrame) -> pd.Series:
        """Return character count of PR body, treating null as empty string.

        Args:
            df: DataFrame with a 'body' column.

        Returns:
            pd.Series of integer character counts, one per row.
        """
        return df["body"].fillna("").astype(str).str.len()

    def _build_cat_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a cleaned categorical DataFrame ready for OHE.

        Fills nulls with MISSING, casts to str, collapses rare categories
        (using self._rare_categories from fit) into UNKNOWN, then replaces
        any remaining unseen values with UNKNOWN.

        Args:
            df: Source DataFrame.

        Returns:
            DataFrame with one column per categorical feature, dtype object.
        """
        result: dict[str, pd.Series] = {}
        for col in CATEGORICAL_COLS:
            vals = df[col].fillna(_MISSING).astype(str)
            rare = self._rare_categories.get(col)
            if rare:
                vals = vals.where(~vals.isin(rare), other=_UNKNOWN)
            known = self._cat_known_values.get(col)
            if known is not None:
                vals = vals.where(vals.isin(known), other=_UNKNOWN)
            result[col] = vals
        return pd.DataFrame(result, index=df.index)

    @staticmethod
    def _all_numeric_cols() -> list[str]:
        """Return the ordered list of all numeric feature names including body_length."""
        return NUMERIC_COLS + ["body_length"]

    def _build_feature_names(self) -> list[str]:
        """Build the complete ordered list of output column names.

        Called at the end of fit() once all transformers are ready.

        Returns:
            Flat list of feature name strings in the same column order as
            the array returned by transform().
        """
        numeric_names = [f"log1p_{c}" for c in self._all_numeric_cols()]
        ohe_names = self._ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
        flag_names = ["touches_ci", "touches_tests", "touches_config"]
        bool_names = ["has_plan"]
        submission_bool_names = [
            "is_experimental_submission",
            "empty_description",
        ]
        cyclic_names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        entropy_names = ["file_entropy"]
        return (
            numeric_names
            + ohe_names
            + flag_names
            + bool_names
            + submission_bool_names
            + cyclic_names
            + entropy_names
        )
