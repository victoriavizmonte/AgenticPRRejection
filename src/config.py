"""
Configuration file for Agentic PR Rejection Prediction thesis.

This module contains all paths, hyperparameters, and settings used across the project.

Applicability:
  - Applicable to all models.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they do not exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
HF_REPO_ID = "hao-li/AIDev"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "pr_closed.parquet"
MIN_STARS = 500

# Multimodal LR (Model 1) configuration
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768  # CodeBERT embedding dimension

# Training configuration
RANDOM_SEED = 42

# Logistic Regression hyperparameters
LR_MAX_ITER = 5000
LR_C = 1.0

# Processing configuration
BATCH_SIZE = 32  # For embedding extraction

# Device configuration (auto-detect)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
