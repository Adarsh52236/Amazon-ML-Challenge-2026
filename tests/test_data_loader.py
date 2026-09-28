"""Unit tests for data loading module."""

from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.data_loader import (
    load_tabular_file,
    load_ground_truth,
    discover_dataset_files,
    EXPECTED_SOURCE_COLUMNS,
    EXPECTED_GROUND_TRUTH_COLUMNS,
)


def test_load_tabular_tsv(tmp_path):
    """Test loading valid TSV and verify column preservation."""
    sample_tsv = tmp_path / "sample.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00123\tAcme Corp\t123 Main St, Springfield\tUS\n"
        "S1-00456\tBeta LLC\t\tIndia\n"
    )
    sample_tsv.write_text(content, encoding="utf-8")

    df = load_tabular_file(sample_tsv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == EXPECTED_SOURCE_COLUMNS
    # Verify string preservation of leading zero IDs
    assert df.loc[0, "entity_id"] == "S1-00123"
    assert isinstance(df.loc[0, "entity_id"], str)
    # Verify missing value preservation (empty string preserved without row drop)
    assert df.loc[1, "business_address"] == ""
    assert len(df) == 2


def test_string_id_numeric_preservation(tmp_path):
    """Test that pure numeric IDs with leading zeros are strictly preserved as strings."""
    sample_tsv = tmp_path / "numeric_ids.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "00001\tTest One\tAddress 1\tUS\n"
        "00099\tTest Two\tAddress 2\tIndia\n"
    )
    sample_tsv.write_text(content, encoding="utf-8")

    df = load_tabular_file(sample_tsv)
    assert df.loc[0, "entity_id"] == "00001"
    assert df.loc[1, "entity_id"] == "00099"
    assert isinstance(df.loc[0, "entity_id"], str)
    assert str(df["entity_id"].dtype) in ("object", "string", "str")


def test_missing_file_raises_error(tmp_path):
    """Test loading nonexistent file raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.tsv"
    with pytest.raises(FileNotFoundError):
        load_tabular_file(missing)


def test_discover_dataset_files(tmp_path):
    """Test discovering all official files in a directory layout."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    (train_dir / "train_source1.tsv").write_text("header\n", encoding="utf-8")
    (train_dir / "train_source2.tsv").write_text("header\n", encoding="utf-8")
    (train_dir / "train_ground_truth.tsv").write_text("header\n", encoding="utf-8")
    (test_dir / "test_source1.tsv").write_text("header\n", encoding="utf-8")

    discovered = discover_dataset_files(tmp_path)
    assert "train_source1" in discovered
    assert "train_source2" in discovered
    assert "train_ground_truth" in discovered
    assert "test_source1" in discovered
    assert "test_source2" not in discovered


def test_load_ground_truth_dict(tmp_path):
    """Test loading ground truth as parsed dictionary."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    gt_file = train_dir / "train_ground_truth.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-10, S3-20\n"
        "S1-2\t\n"
        "S1-3\tS2-30\n"
    )
    gt_file.write_text(content, encoding="utf-8")

    mapping = load_ground_truth(root_dir=tmp_path, as_dict=True)
    assert mapping["S1-1"] == ["S2-10", "S3-20"]
    assert mapping["S1-2"] == []
    assert mapping["S1-3"] == ["S2-30"]
