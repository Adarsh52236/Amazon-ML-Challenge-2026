"""Unit tests for configuration module."""

import os
from pathlib import Path
import pytest

from business_entity_resolution.config import (
    DEFAULT_DATASET_ROOT,
    ENV_DATASET_ROOT_KEY,
    DatasetNotFoundError,
    get_dataset_root,
    get_train_dir,
    get_test_dir,
    get_source_path,
    get_ground_truth_path,
)


def test_default_path_handling(monkeypatch):
    """Test default path behavior when env var is cleared."""
    monkeypatch.delenv(ENV_DATASET_ROOT_KEY, raising=False)
    # If DEFAULT_DATASET_ROOT exists on this machine, it should resolve to it
    if DEFAULT_DATASET_ROOT.exists():
        assert get_dataset_root() == DEFAULT_DATASET_ROOT.resolve()


def test_env_var_override(tmp_path, monkeypatch):
    """Test path resolution through environment variable."""
    fake_dataset = tmp_path / "mock_dataset"
    fake_dataset.mkdir()
    monkeypatch.setenv(ENV_DATASET_ROOT_KEY, str(fake_dataset))
    assert get_dataset_root() == fake_dataset.resolve()


def test_explicit_path_override(tmp_path):
    """Test explicit path argument taking precedence over all else."""
    custom_path = tmp_path / "custom_data"
    custom_path.mkdir()
    assert get_dataset_root(override_path=custom_path) == custom_path.resolve()


def test_missing_directory_raises_error(tmp_path):
    """Test that nonexistent path raises informative DatasetNotFoundError."""
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(DatasetNotFoundError) as exc_info:
        get_dataset_root(override_path=nonexistent)
    assert "Dataset root directory not found" in str(exc_info.value)
    assert ENV_DATASET_ROOT_KEY in str(exc_info.value)


def test_subfolder_navigation(tmp_path):
    """Test resolving train/test subfolders when organized under 'dataset'."""
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Create dummy source and ground truth files
    (train_dir / "train_source1.tsv").write_text("header\n", encoding="utf-8")
    (train_dir / "train_ground_truth.tsv").write_text("header\n", encoding="utf-8")

    assert get_train_dir(override_path=tmp_path) == train_dir.resolve()
    assert get_test_dir(override_path=tmp_path) == test_dir.resolve()
    assert get_source_path("train", 1, override_path=tmp_path) == (train_dir / "train_source1.tsv").resolve()
    assert get_ground_truth_path(override_path=tmp_path) == (train_dir / "train_ground_truth.tsv").resolve()
