"""
CodeBERT embedding extraction for Multimodal LR (Model 1).

Given a DataFrame of diff chunks (output of features.build_diff_chunks),
this module produces a single mean-pooled 768-dim embedding per PR 
using frozen CodeBERT, and save/load the resulting numpy arrays to disk.

Design decisions:
  - Frozen model: eval() mode, all parameters have requires_grad=False.
  - Representation: [CLS] token from the last hidden state per chunk.
  - Pooling: mean across all chunks for a PR; zero vector if no chunks.
  - Tokenization: max_length=512, truncation=True, padding=True per chunk.
  - Cache: extraction is skipped if output files already exist on disk.
  - Saved files per split: {split}_embeddings.npy, {split}_labels.npy,
    {split}_pr_ids.npy — all in the same output_dir.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from config import (
    BATCH_SIZE,
    CODEBERT_MODEL_NAME,
    DEVICE,
    EMBEDDING_DIM,
    RAW_DATA_DIR,
)

CHECKPOINT_EVERY: int = 500  # Save checkpoint after every N PRs processed

logger = logging.getLogger(__name__)


# Model loading
def load_codebert(
    model_name: str = CODEBERT_MODEL_NAME,
    device: str = DEVICE,
) -> tuple[AutoTokenizer, AutoModel]:
    """Load frozen CodeBERT tokenizer and model.

    Args:
        model_name: HuggingFace model identifier. Default is
            'microsoft/codebert-base'.
        device: Target device ('cuda', 'mps', or 'cpu').

    Returns:
        A (tokenizer, model) tuple. The model is in eval mode with all
        parameters frozen.
    """
    logger.info("Loading CodeBERT tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)
    logger.info("CodeBERT loaded on device: %s", device)
    return tokenizer, model


# Single-PR embedding
def embed_chunks(
    chunks: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed a list of diff chunks and return a mean-pooled 768-dim vector.

    Each chunk is tokenized independently and encoded with CodeBERT. The
    [CLS] token embedding is extracted from the last hidden state for each
    chunk. The final representation is the mean across all chunks.

    Args:
        chunks: List of diff hunk strings for one PR. May be empty.
        tokenizer: CodeBERT tokenizer.
        model: Frozen CodeBERT model.
        device: Device to run inference on.
        batch_size: Number of chunks to process per forward pass.

    Returns:
        1-D float32 numpy array of shape (768,). Returns a zero vector if
        chunks is empty.
    """
    if not chunks:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    cls_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            encoded = tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # [CLS] token: index 0 of the sequence dimension
            cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            cls_embeddings.append(cls_vecs)

    all_cls = np.concatenate(cls_embeddings, axis=0)  # shape (n_chunks, 768)
    return all_cls.mean(axis=0).astype(np.float32)  # shape (768,)


# Split-level extraction and caching
def extract_and_save_embeddings(
    diff_chunks_df: pd.DataFrame,
    split_name: str,
    output_dir: Path,
    model_name: str = CODEBERT_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
    checkpoint_dir: Path | None = None,
) -> np.ndarray:
    """Extract CodeBERT embeddings for all PRs in one split and save to disk.

    If the output files already exist, they are loaded from disk and returned
    without re-running inference (cache behaviour).

    Checkpoints are written to checkpoint_dir as {split_name}_checkpoint.npz
    every CHECKPOINT_EVERY PRs. On startup the function detects an existing
    checkpoint and resumes from the last saved position. The checkpoint file is
    deleted automatically after the final .npy files are written.

    Args:
        diff_chunks_df: DataFrame produced by features.build_diff_chunks.
            Must contain columns: pr_id, label, diff_chunks. The DataFrame
            index must be 0-based and sequential (standard usage).
        split_name: One of 'train', 'val', 'test'. Used for file naming.
        output_dir: Directory where .npy files are written.
        model_name: HuggingFace model identifier for CodeBERT.
        batch_size: Chunks per forward pass.
        device: Inference device.
        checkpoint_dir: Directory for checkpoint files. Defaults to
            RAW_DATA_DIR. When None, RAW_DATA_DIR from config is used.

    Returns:
        Embedding matrix as float32 numpy array of shape (n_prs, 768).

    Side effects:
        Writes three files to output_dir: {split_name}_embeddings.npy (float32),
        {split_name}_labels.npy (int32), and {split_name}_pr_ids.npy. A
        checkpoint file in checkpoint_dir is deleted automatically on successful
        completion.

    Raises:
        KeyError: If diff_chunks_df is missing required columns.
    """
    for col in ("pr_id", "label", "diff_chunks"):
        if col not in diff_chunks_df.columns:
            raise KeyError(
                f"Column '{col}' not found in diff_chunks_df. "
                f"Available columns: {list(diff_chunks_df.columns)}"
            )

    emb_path = output_dir / f"{split_name}_embeddings.npy"
    lbl_path = output_dir / f"{split_name}_labels.npy"
    ids_path = output_dir / f"{split_name}_pr_ids.npy"

    if emb_path.exists() and lbl_path.exists():
        logger.info(
            "Cache found for split '%s'. Loading from %s.", split_name, output_dir
        )
        return np.load(emb_path)

    # Checkpoint setup
    if checkpoint_dir is None:
        checkpoint_dir = RAW_DATA_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{split_name}_checkpoint.npz"

    n_prs = len(diff_chunks_df)

    if checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        embeddings = ckpt["embeddings"].copy()
        completed_count = int(ckpt["completed_count"])
        logger.info(
            "Resuming from checkpoint: %d/%d PRs already embedded.",
            completed_count,
            n_prs,
        )
    else:
        embeddings = np.zeros((n_prs, EMBEDDING_DIM), dtype=np.float32)
        completed_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_codebert(model_name=model_name, device=device)

    logger.info("Extracting embeddings for %d PRs (split='%s').", n_prs, split_name)

    df_remaining = diff_chunks_df.iloc[completed_count:]
    rows_done = 0

    for pos, (_, row) in enumerate(tqdm(
        df_remaining.iterrows(),
        total=n_prs,
        initial=completed_count,
        desc=f"Embedding [{split_name}]",
    )):
        embeddings[completed_count + pos] = embed_chunks(
            chunks=row["diff_chunks"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        rows_done += 1
        global_count = completed_count + rows_done

        if global_count % CHECKPOINT_EVERY == 0:
            np.savez(checkpoint_path, embeddings=embeddings, completed_count=global_count)
            logger.info("Checkpoint saved at %d/%d PRs.", global_count, n_prs)

    labels = diff_chunks_df["label"].to_numpy(dtype=np.int32)
    pr_ids = diff_chunks_df["pr_id"].to_numpy()

    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    np.save(ids_path, pr_ids)

    logger.info(
        "Saved embeddings (%s), labels (%s), pr_ids (%s) to %s.",
        embeddings.shape,
        labels.shape,
        pr_ids.shape,
        output_dir,
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint deleted after successful completion.")

    return embeddings


# Loading cached embeddings
def load_embeddings(
    split_name: str,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-saved embeddings and labels from disk.

    Args:
        split_name: One of 'train', 'val', 'test'.
        output_dir: Directory containing the .npy files.

    Returns:
        Tuple of (embeddings, labels). embeddings is float32 of shape
        (n_prs, 768); labels is int32 of shape (n_prs,). The companion
        {split_name}_pr_ids.npy file is not loaded by this function.

    Raises:
        FileNotFoundError: If the expected files are not found in output_dir.
    """
    emb_path = output_dir / f"{split_name}_embeddings.npy"
    lbl_path = output_dir / f"{split_name}_labels.npy"

    for path in (emb_path, lbl_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}. "
                "Run extract_and_save_embeddings first."
            )

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)
    logger.info(
        "Loaded %s embeddings: shape %s, labels: shape %s.",
        split_name,
        embeddings.shape,
        labels.shape,
    )
    return embeddings, labels
