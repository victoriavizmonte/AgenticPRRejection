"""
Text feature extraction for Agentic PR Rejection prediction.

This module encodes PR title and body text into dense sentence embeddings
using a frozen sentence-transformers model. This module must never use
post-submission fields (e.g., review comments, post-closure metadata).

Applicability:
  - Applicable to Multimodal LR only. Standard LLM and reasoning LLM classifiers
    handle title and body text directly in their own prompt-formatting functions
    and do not import from this module.

Design decisions:
  - Model: all-MiniLM-L6-v2 (384-dimensional, lightweight)
  - Input: title and body columns from the pull_request table
  - Separator: ' [SEP] ' between title and body to signal field boundary
  - Null handling: None or NaN in either field is replaced with empty string
  - Interface: sklearn-style fit / transform / fit_transform for compatibility
    with permutation importance and pipeline toggling
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384


class PRTextEncoder:
    """Sentence-embedding encoder for PR title and body text.

    Encodes each PR as a single 384-dimensional vector by concatenating
    title and body with a [SEP] separator token and passing the result
    through a frozen all-MiniLM-L6-v2 model.

    Follows the sklearn fit / transform / fit_transform interface so it
    can be composed with other feature transformers and used directly in
    permutation importance calculations.

    Attributes:
        model_name: HuggingFace model identifier for the sentence encoder.
        batch_size: Number of texts encoded per forward pass.
        device: Torch device string (e.g. 'cuda', 'cpu'). If None, the
            sentence-transformers library selects the device automatically.

    Example:
        >>> encoder = PRTextEncoder()
        >>> X_train = encoder.fit_transform(train_df[["title", "body"]])
        >>> X_val = encoder.transform(val_df[["title", "body"]])
        >>> assert X_train.shape[1] == 384
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the encoder configuration without loading the model.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                all-MiniLM-L6-v2, which produces 384-dimensional vectors.
            batch_size: Number of texts per encoding batch. Larger values
                use more GPU memory but reduce total forward-pass overhead.
                Defaults to 64.
            device: Torch device string passed to SentenceTransformer.
                If None, the library auto-selects (CUDA > MPS > CPU).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PRTextEncoder":
        """Load the sentence-transformer model.

        No training occurs because the model weights are frozen. This method
        exists to satisfy the sklearn interface and to defer the (one-time)
        model download until the encoder is actually used.

        Args:
            df: DataFrame containing at least 'title' and 'body' columns.
                The DataFrame is not modified; it is accepted here only for
                interface consistency with sklearn transformers.

        Returns:
            self, to allow method chaining (encoder.fit(df).transform(df)).

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence encoder: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._fitted = True
        logger.info("Sentence encoder loaded.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode PR title and body into sentence embeddings.

        Concatenates title and body with ' [SEP] ' before encoding.
        Null or empty fields are treated as empty strings so that every
        row produces a valid (non-NaN) embedding vector.

        Args:
            df: DataFrame containing 'title' and 'body' columns.
                May contain NaN or None values in either column.

        Returns:
            np.ndarray of shape (n_samples, 384) and dtype float32.
            Row i corresponds to row i in df.

        Raises:
            RuntimeError: If fit() has not been called first.
            KeyError: If 'title' or 'body' columns are absent from df.
        """
        if not self._fitted:
            raise RuntimeError(
                "PRTextEncoder.fit() must be called before transform(). "
                "Use fit_transform() to do both in one step."
            )
        for col in ("title", "body"):
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

        texts = self._build_texts(df)
        logger.info(
            "Encoding %d PR texts with model '%s' (batch_size=%d).",
            len(texts),
            self.model_name,
            self.batch_size,
        )
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("Encoding complete. Output shape: %s.", embeddings.shape)
        return embeddings

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Load the model and encode in one call.

        Equivalent to fit(df).transform(df). Provided for sklearn
        compatibility and convenience.

        Args:
            df: DataFrame containing 'title' and 'body' columns.

        Returns:
            np.ndarray of shape (n_samples, 384) and dtype float32.
        """
        return self.fit(df).transform(df)

    @staticmethod
    def _build_texts(df: pd.DataFrame) -> list[str]:
        """Concatenate title and body into a single string per PR.

        Null values in either column are replaced with an empty string
        before concatenation. The separator token ' [SEP] ' is inserted
        between the two fields to signal the field boundary to the encoder.

        Args:
            df: DataFrame with 'title' and 'body' columns.

        Returns:
            List of n_samples strings, one per row in df.
        """
        titles = df["title"].fillna("").astype(str)
        bodies = df["body"].fillna("").astype(str)
        return (titles + " [SEP] " + bodies).tolist()
