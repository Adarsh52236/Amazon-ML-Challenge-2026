"""Centralized configuration module for Business Entity Resolution.

Provides environment-aware path discovery, Windows compatibility, and robust
validation of dataset locations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Attempt to load environment variables from a local .env file if available
try:
    from dotenv import load_dotenv

    # Search for .env starting at the current working directory or repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    dotenv_path = repo_root / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Project directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Default dataset root path on local developer environment
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\adars\Downloads\business_entity_resolution\student_resource"
)

# Environment variable name for dataset path override
ENV_DATASET_ROOT_KEY = "BUSINESS_ENTITY_DATASET_ROOT"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the official dataset directory cannot be found."""

    pass


def get_dataset_root(override_path: Optional[Path | str] = None) -> Path:
    """Resolve the dataset root directory following configuration priority:
    1. Explicit function argument (override_path)
    2. Environment variable (BUSINESS_ENTITY_DATASET_ROOT)
    3. Default path (C:\\Users\\adars\\Downloads\\business_entity_resolution\\student_resource)
    4. Local fallback (./data/official if present)

    Parameters
    ----------
    override_path : Optional[Path | str]
        Optional explicit path to override all defaults.

    Returns
    -------
    Path
        Resolved dataset root path.

    Raises
    ------
    DatasetNotFoundError
        If the resolved path does not exist on the filesystem.
    """
    candidate_source = "default"
    if override_path is not None:
        path = Path(override_path)
        candidate_source = "explicit argument"
    elif os.getenv(ENV_DATASET_ROOT_KEY):
        path = Path(os.getenv(ENV_DATASET_ROOT_KEY).strip())
        candidate_source = f"environment variable ({ENV_DATASET_ROOT_KEY})"
    elif DEFAULT_DATASET_ROOT.exists():
        path = DEFAULT_DATASET_ROOT
        candidate_source = "default developer path"
    elif (DATA_DIR / "official").exists():
        path = DATA_DIR / "official"
        candidate_source = "local data/official fallback"
    else:
        path = DEFAULT_DATASET_ROOT
        candidate_source = "default developer path (not found)"

    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise DatasetNotFoundError(
            f"Dataset root directory not found at: '{resolved_path}'\n"
            f"Resolution source: {candidate_source}\n"
            f"Please ensure the dataset is present or configure the correct path using:\n"
            f"  1. Setting the environment variable '{ENV_DATASET_ROOT_KEY}' in your terminal or .env file\n"
            f"  2. Passing an explicit override_path argument to get_dataset_root()\n"
            f"  3. Placing the dataset inside '{DATA_DIR / 'official'}'"
        )

    return resolved_path


def get_data_base_dir(override_path: Optional[Path | str] = None) -> Path:
    """Return the base directory containing the 'train' and 'test' subdirectories.

    Handles both layouts:
    - Root containing a 'dataset' subfolder (e.g. student_resource/dataset/train)
    - Root directly containing 'train' and 'test' (e.g. data/official/train)
    """
    root = get_dataset_root(override_path)
    nested_dataset_dir = root / "dataset"
    if nested_dataset_dir.is_dir() and (nested_dataset_dir / "train").is_dir():
        return nested_dataset_dir
    return root


def get_train_dir(override_path: Optional[Path | str] = None) -> Path:
    """Return the path to the training dataset folder."""
    base = get_data_base_dir(override_path)
    train_dir = base / "train"
    if not train_dir.exists():
        raise DatasetNotFoundError(
            f"Train directory not found at: '{train_dir}'. "
            f"Expected inside base dataset directory: '{base}'"
        )
    return train_dir


def get_test_dir(override_path: Optional[Path | str] = None) -> Path:
    """Return the path to the test dataset folder."""
    base = get_data_base_dir(override_path)
    test_dir = base / "test"
    if not test_dir.exists():
        raise DatasetNotFoundError(
            f"Test directory not found at: '{test_dir}'. "
            f"Expected inside base dataset directory: '{base}'"
        )
    return test_dir


def get_source_path(
    split: str,
    source_num: int,
    override_path: Optional[Path | str] = None,
) -> Path:
    """Get the path to a specific source TSV file.

    Parameters
    ----------
    split : str
        Dataset split: 'train' or 'test'.
    source_num : int
        Source identifier: 1, 2, or 3.
    override_path : Optional[Path | str]
        Optional dataset root path override.

    Returns
    -------
    Path
        Path to the requested source TSV file.
    """
    split = split.lower().strip()
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")
    if source_num not in {1, 2, 3}:
        raise ValueError(
            f"Invalid source_num {source_num}. Expected 1, 2, or 3."
        )

    split_dir = get_train_dir(override_path) if split == "train" else get_test_dir(override_path)
    filename = f"{split}_source{source_num}.tsv"
    file_path = split_dir / filename
    if not file_path.exists():
        raise DatasetNotFoundError(f"Source file not found at: '{file_path}'")
    return file_path


def get_ground_truth_path(override_path: Optional[Path | str] = None) -> Path:
    """Get the path to the training ground truth file."""
    train_dir = get_train_dir(override_path)
    gt_path = train_dir / "train_ground_truth.tsv"
    if not gt_path.exists():
        raise DatasetNotFoundError(
            f"Ground truth file not found at: '{gt_path}'"
        )
    return gt_path


def configure_console_encoding() -> None:
    """Ensure standard output and error use UTF-8 on Windows consoles.

    Prevents charmap/cp1252 UnicodeEncodeError when printing international
    entity names or addresses.
    """
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
