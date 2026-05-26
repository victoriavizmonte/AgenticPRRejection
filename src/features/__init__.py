"""Feature extraction package for Agentic PR Rejection prediction.

Submodules:
    diff_features      : Hunk-based diff chunking for CodeBERT encoding.
    text_features      : Sentence-embedding encoder for PR title and body text.
    metadata_features  : Structured metadata encoder (numeric, categorical, flags,
                         cyclic time, entropy).

Applicability:
  - Package-level exports (build_diff_chunks, PRMetadataEncoder, etc.):
    applicable to Multimodal LR only.
  - Standard LLM and reasoning LLM classifiers import submodules directly
    (e.g., features.metadata_features) rather than using this package's exports.
"""

from features.diff_features import (
    build_diff_chunks,
    drop_empty_chunks,
    assert_no_empty_chunks,
)
from features.metadata_features import PRMetadataEncoder

__all__ = [
    "build_diff_chunks",
    "drop_empty_chunks",
    "assert_no_empty_chunks",
    "PRMetadataEncoder",
]
