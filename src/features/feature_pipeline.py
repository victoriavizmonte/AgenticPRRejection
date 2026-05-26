"""
Feature pipeline orchestrator for Agentic PR Rejection prediction.

This module accepts multiple named feature groups (diff embeddings, text
embeddings, metadata), fits each encoder on training data only, and concatenates
the active groups into a single feature matrix.

Applicability:
  - Applicable to Multimodal LR only. Standard LLM and reasoning LLM classifiers
    use raw PR data formatted directly as text prompts and do not use this
    pipeline.

Design decisions:
  - Canonical group order is ["diff", "text", "metadata"]. Groups are always
    concatenated in this order regardless of the order given in active_features.
  - The "diff" group uses pre-computed CodeBERT embeddings (extracted on GPU
    via model1_embedder). No encoder is needed; the array is passed directly.
  - "text" and "metadata" groups use sklearn-compatible encoders with
    fit / transform methods.
  - group_slices_ maps each active group name to the slice of columns it
    occupies in the concatenated output.
  - The pipeline is saved as an output via joblib for checkpoint persistence.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_ORDER: list[str] = ["diff", "text", "metadata"]


class FeaturePipeline:
    """Orchestrate multi-group feature encoding and concatenation.

    Accepts up to three named feature groups — diff (CodeBERT embeddings),
    text (sentence embeddings), and metadata (structured features) — and
    concatenates the active subset into a single (n_samples, total_dim)
    matrix. Groups are always in canonical order: diff, text, metadata.

    The pipeline exposes group_slices_ after fitting so that downstream code
    can address each group's columns precisely.

    Attributes:
        active_features: Ordered list of active group names (subset of
            CANONICAL_ORDER). Groups are concatenated in canonical order.
        group_slices_: Dict mapping each active group name to its column
            slice in the output matrix. Populated by fit().

    Example:
        >>> from features.text_features import PRTextEncoder
        >>> from features.metadata_features import PRMetadataEncoder
        >>> pipeline = FeaturePipeline(
        ...     active_features=["diff", "text", "metadata"],
        ...     text_encoder=PRTextEncoder(),
        ...     metadata_encoder=PRMetadataEncoder(),
        ... )
        >>> pipeline.fit(train_df, X_diff_train=X_diff_train)
        >>> X_train = pipeline.transform(train_df, X_diff=X_diff_train)
        >>> X_test  = pipeline.transform(test_df,  X_diff=X_diff_test)
    """

    def __init__(
        self,
        active_features: list[str],
        text_encoder=None,
        metadata_encoder=None,
    ) -> None:
        """Initialize the pipeline without fitting any encoder.

        Args:
            active_features: List of group names to include. Must be a
                non-empty subset of ["diff", "text", "metadata"].
            text_encoder: A fitted-or-unfitted encoder with fit(df) and
                transform(df) methods producing a float ndarray of shape
                (n_samples, d_text). Required when "text" is active.
            metadata_encoder: A fitted-or-unfitted encoder with fit(df),
                transform(df), and feature_names_ attribute. Required when
                "metadata" is active.

        Raises:
            ValueError: If active_features is empty.
            ValueError: If any name in active_features is not in
                CANONICAL_ORDER.
            ValueError: If "text" is active but text_encoder is None.
            ValueError: If "metadata" is active but metadata_encoder is None.
        """
        if not active_features:
            raise ValueError(
                "active_features must contain at least one group name. "
                f"Valid options: {CANONICAL_ORDER}"
            )

        unknown = [f for f in active_features if f not in CANONICAL_ORDER]
        if unknown:
            raise ValueError(
                f"Unknown feature groups: {unknown}. "
                f"Valid options: {CANONICAL_ORDER}"
            )

        if "text" in active_features and text_encoder is None:
            raise ValueError(
                "'text' is listed in active_features but text_encoder is None. "
                "Provide a PRTextEncoder instance."
            )

        if "metadata" in active_features and metadata_encoder is None:
            raise ValueError(
                "'metadata' is listed in active_features but metadata_encoder is None. "
                "Provide a PRMetadataEncoder instance."
            )

        # Store in canonical order so the rest of the code can rely on it.
        self.active_features: list[str] = [
            g for g in CANONICAL_ORDER if g in active_features
        ]
        self._text_encoder = text_encoder
        self._metadata_encoder = metadata_encoder
        self.group_slices_: dict[str, slice] = {}
        self._fitted: bool = False

    # Public interface
    def fit(
        self,
        df_train: pd.DataFrame,
        X_diff_train: Optional[np.ndarray] = None,
    ) -> "FeaturePipeline":
        """Fit encoders on training data and compute group_slices_.

        The "diff" group requires no fitting (embeddings are pre-computed).
        For "text", the encoder's fit() method loads the frozen
        sentence-transformer. For "metadata", fit() computes medians and
        fits the StandardScaler and OneHotEncoder on training statistics.

        group_slices_ is populated after fitting based on each encoder's
        actual output dimension, determined by a single forward pass on a
        one-row sample. This avoids hardcoding dimension constants.

        Args:
            df_train: Training DataFrame. Must contain all columns required
                by the active encoders.
            X_diff_train: Pre-computed CodeBERT diff embeddings for the
                training set. Shape (n_train, d_diff). Required when "diff"
                is in active_features.

        Returns:
            self, to allow method chaining.

        Raises:
            ValueError: If "diff" is active and X_diff_train is None.
            ValueError: If X_diff_train has fewer than 1 row.
        """
        if "diff" in self.active_features:
            if X_diff_train is None:
                raise ValueError(
                    "'diff' is in active_features but X_diff_train is None. "
                    "Pass the pre-computed CodeBERT embedding array."
                )
            if X_diff_train.shape[0] == 0:
                raise ValueError(
                    "X_diff_train must contain at least one row."
                )

        if "text" in self.active_features:
            logger.info("Fitting text encoder on training data.")
            self._text_encoder.fit(df_train)
            logger.info("Text encoder fitted.")

        if "metadata" in self.active_features:
            logger.info("Fitting metadata encoder on training data.")
            self._metadata_encoder.fit(df_train)
            logger.info("Metadata encoder fitted.")

        self._compute_slices(df_train, X_diff_train)
        self._fitted = True
        logger.info(
            "FeaturePipeline fitted. group_slices_: %s",
            {k: (v.start, v.stop) for k, v in self.group_slices_.items()},
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        X_diff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode df and concatenate active feature groups.

        Args:
            df: DataFrame to transform. Must contain all columns required
                by the active encoders.
            X_diff: Pre-computed CodeBERT diff embeddings for this split.
                Shape (n_samples, d_diff). Required when "diff" is active.

        Returns:
            np.ndarray of shape (n_samples, total_dim). Column order matches
            the canonical group order (diff, text, metadata).

        Raises:
            RuntimeError: If fit() has not been called first.
            ValueError: If "diff" is active and X_diff is None.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeaturePipeline.fit() must be called before transform()."
            )

        if "diff" in self.active_features and X_diff is None:
            raise ValueError(
                "'diff' is in active_features but X_diff is None. "
                "Pass the pre-computed CodeBERT embedding array."
            )

        parts: list[np.ndarray] = []
        for group in CANONICAL_ORDER:
            if group not in self.active_features:
                continue
            if group == "diff":
                parts.append(X_diff)
            elif group == "text":
                parts.append(self._text_encoder.transform(df))
            elif group == "metadata":
                parts.append(self._metadata_encoder.transform(df))

        return np.concatenate(parts, axis=1)

    # Private helpers
    def _compute_slices(
        self,
        df_train: pd.DataFrame,
        X_diff_train: Optional[np.ndarray],
    ) -> None:
        """Compute group_slices_ from actual encoder output dimensions.

        Encodes a single training row for text and metadata to obtain the
        true output dimension without hardcoding constants.

        Args:
            df_train: Training DataFrame (used for 1-row sample encoding).
            X_diff_train: Pre-computed diff embeddings for training.
        """
        offset = 0
        sample_df = df_train.iloc[:1]

        for group in CANONICAL_ORDER:
            if group not in self.active_features:
                continue

            if group == "diff":
                dim = X_diff_train.shape[1]
            elif group == "text":
                sample_out = self._text_encoder.transform(sample_df)
                dim = sample_out.shape[1]
            elif group == "metadata":
                dim = len(self._metadata_encoder.feature_names_)

            self.group_slices_[group] = slice(offset, offset + dim)
            offset += dim
