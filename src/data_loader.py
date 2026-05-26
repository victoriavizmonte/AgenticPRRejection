"""
Data loader for AIDev dataset.

This module handles downloading and loading raw parquet tables with automatic environment
detection. Supports execution environments:

    Local:  Reads from RAW_DATA_DIR defined in config.py.
            Downloads from HuggingFace on first run via hf_hub_download.
    Colab:  Mounts Google Drive and reads from COLAB_DRIVE_DIR.
            No download — tables must already exist in the shared Drive
            folder. See README.md for data setup instructions.

Environment is detected automatically; no code changes needed when switching.

Applicability:
  - Applicable to all models.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from config import RAW_DATA_DIR, HF_REPO_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_colab_drive_dir() -> Path:
    """Resolve raw data directory from DRIVE_BASE if set, else use default."""
    drive_base = os.environ.get('DRIVE_BASE')
    if drive_base:
        return Path(drive_base) / 'data' / 'raw'
    return Path('/content/drive/MyDrive/Thesis/AgenticPRRejection/data/raw')
COLAB_DRIVE_DIR = _get_colab_drive_dir()

TABLE_NAMES = [
    "pull_request",
    "repository",
    "pr_commits",
    "pr_commit_details",
    "pr_task_type",
    "related_issue",
]


# Environment detection
def is_colab() -> bool:
    """Return True if running inside a Google Colab notebook."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_environment() -> str:
    """Return a string label for the current execution environment."""
    if is_colab():
        return "colab"
    return "local"


def get_table_path(name: str) -> Path:
    """
    Resolve the path for a given table based on the current environment.

    Args:
        name: Table name (e.g. 'pull_request').

    Returns:
        Absolute path to the parquet file.
    """
    env = get_environment()
    if env == "colab":
        return COLAB_DRIVE_DIR / f"{name}.parquet"
    return RAW_DATA_DIR / f"{name}.parquet"


# Colab Drive mount helper
def _ensure_drive_mounted() -> None:
    """Mount Google Drive if running in Colab and Drive is not yet mounted."""
    mount_point = Path("/content/drive")
    if not (mount_point / "MyDrive").exists():
        logger.info("Mounting Google Drive...")
        from google.colab import drive
        drive.mount(str(mount_point))
    else:
        logger.info("Google Drive already mounted.")


# Public API
def download_raw_tables(force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load raw AIDev parquet tables, downloading from HuggingFace if needed.

    Behaviour by environment:
        Local  : Downloads via hf_hub_download on first run; caches locally.
        Colab  : Mounts Drive and reads from COLAB_DRIVE_DIR (no download).
                 Tables must already exist in the shared Drive folder. See README.md for setup instructions.

    Args:
        force: Re-download even if local cache exists (local env only).

    Returns:
        Dict mapping table name to its DataFrame.

    Raises:
        FileNotFoundError: On Colab if tables are missing from
                           the expected location.
    """
    env = get_environment()
    logger.info(f"Environment: {env}")

    if env == "colab":
        _ensure_drive_mounted()
        return _load_from_paths(
            {name: COLAB_DRIVE_DIR / f"{name}.parquet" for name in TABLE_NAMES},
            setup_hint="Ensure the shared Drive folder is linked at the correct DRIVE_BASE path. See README.md for setup instructions."
        )

    # Local: download from HuggingFace if not cached
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}

    for name in TABLE_NAMES:
        local_path = RAW_DATA_DIR / f"{name}.parquet"

        if local_path.exists() and not force:
            logger.info(f"Loading cached {name} from {local_path}")
            tables[name] = pd.read_parquet(local_path)
        else:
            logger.info(f"Downloading {name} from HuggingFace ({HF_REPO_ID})")
            tmp_path = Path(hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{name}.parquet",
                repo_type="dataset",
                local_dir=str(RAW_DATA_DIR),
                force_download=force,
            ))
            if tmp_path != local_path:
                shutil.copy2(tmp_path, local_path)
            tables[name] = pd.read_parquet(local_path)

        logger.info(f"  {name}: {tables[name].shape}")

    return tables


def load_raw_tables(names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load a subset (or all) of raw AIDev tables from the environment-appropriate path.

    Prefer this over download_raw_tables() in notebooks where tables are
    already present, to make the intent explicit.

    Args:
        names: List of table names to load. Defaults to all TABLE_NAMES.

    Returns:
        Dict mapping table name to its DataFrame.

    Raises:
        ValueError: If an unrecognised table name is requested.
        FileNotFoundError: If a table file is missing from the expected path.
    """
    targets = names if names is not None else TABLE_NAMES

    unknown = set(targets) - set(TABLE_NAMES)
    if unknown:
        raise ValueError(f"Unknown table name(s): {unknown}. Valid: {TABLE_NAMES}")

    if get_environment() == "colab":
        _ensure_drive_mounted()

    tables = {}
    for name in targets:
        path = get_table_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Table '{name}' not found at {path}.\n"
                f"  Local : run download_raw_tables() first.\n"
                f"  Colab : ensure the shared Drive folder is linked at the correct DRIVE_BASE path. See README.md for setup instructions.\n"
            )
        tables[name] = pd.read_parquet(path)
        logger.info(f"Loaded {name}: {tables[name].shape}")

    return tables


# Internal helpers
def _load_from_paths(
    paths: Dict[str, Path],
    setup_hint: str,
) -> Dict[str, pd.DataFrame]:
    """Load tables from a pre-resolved path dict with clear error messages."""
    tables = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Table '{name}' not found at {path}.\n  {setup_hint}"
            )
        tables[name] = pd.read_parquet(path)
        logger.info(f"  {name}: {tables[name].shape}")
    return tables


if __name__ == "__main__":
    env = get_environment()
    print(f"Environment : {env}")
    print("Loading raw AIDev tables...\n")
    tables = download_raw_tables()
    for name, df in tables.items():
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"  {name}: {df.shape}  (~{mem_mb:.1f} MB in memory)")