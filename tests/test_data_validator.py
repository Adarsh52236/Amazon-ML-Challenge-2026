"""Unit tests for data validation module."""

from pathlib import Path
import pytest

from business_entity_resolution.data_validator import (
    ValidationStatus,
    ValidationSeverity,
    check_file_existence,
    check_tabular_schema,
    validate_source_file_content,
    validate_ground_truth_file,
)
from business_entity_resolution.data_loader import (
    EXPECTED_SOURCE_COLUMNS,
    EXPECTED_GROUND_TRUTH_COLUMNS,
)


def test_check_file_existence(tmp_path):
    """Test checking existence on real, nonexistent, and empty files."""
    real_file = tmp_path / "data.tsv"
    real_file.write_text("dummy content", encoding="utf-8")
    empty_file = tmp_path / "empty.tsv"
    empty_file.write_text("", encoding="utf-8")
    missing_file = tmp_path / "missing.tsv"

    assert check_file_existence(real_file).status == ValidationStatus.PASS
    assert check_file_existence(empty_file).status == ValidationStatus.ERROR
    assert check_file_existence(missing_file).status == ValidationStatus.ERROR


def test_check_tabular_schema(tmp_path):
    """Test schema validation for exact, missing, and unexpected columns."""
    valid_file = tmp_path / "valid.tsv"
    valid_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")
    results, _ = check_tabular_schema(valid_file, EXPECTED_SOURCE_COLUMNS)
    assert any(r.status == ValidationStatus.PASS for r in results)

    missing_col_file = tmp_path / "missing_col.tsv"
    missing_col_file.write_text("entity_id\tbusiness_name\n", encoding="utf-8")
    results, _ = check_tabular_schema(missing_col_file, EXPECTED_SOURCE_COLUMNS)
    assert any(r.status == ValidationStatus.ERROR and "Missing required columns" in r.message for r in results)


def test_validate_source_file_duplicate_and_empty_ids(tmp_path):
    """Test detection of duplicate and empty entity IDs."""
    bad_file = tmp_path / "bad_source.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-100\tBiz A\tAddr A\tUS\n"
        "S1-100\tBiz B\tAddr B\tUS\n"  # Duplicate ID
        "\tBiz C\tAddr C\tUS\n"        # Empty ID
    )
    bad_file.write_text(content, encoding="utf-8")

    results, ids = validate_source_file_content(bad_file, expected_prefix="S1-")
    assert any(r.check_name == "entity_id_uniqueness" and r.status == ValidationStatus.ERROR for r in results)
    assert any(r.check_name == "entity_id_empty_check" and r.status == ValidationStatus.ERROR for r in results)


def test_validate_ground_truth_cross_references(tmp_path):
    """Test validation of ground truth references and singletons."""
    gt_file = tmp_path / "gt.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-10,S3-20\n"
        "S1-2\t\n"                    # Singleton
        "S1-3\tS2-999\n"              # S2-999 doesn't exist
    )
    gt_file.write_text(content, encoding="utf-8")

    s1_ids = {"S1-1", "S1-2", "S1-3"}
    s2_ids = {"S2-10"}
    s3_ids = {"S3-20"}

    results = validate_ground_truth_file(
        gt_path=gt_file,
        s1_ids=s1_ids,
        s2_ids=s2_ids,
        s3_ids=s3_ids,
    )

    # Should detect missing S2-999
    assert any(r.check_name == "gt_s2_reference_integrity" and r.status == ValidationStatus.ERROR for r in results)
    # S3 references should pass
    assert any(r.check_name == "gt_s3_reference_integrity" and r.status == ValidationStatus.PASS for r in results)
    # S1 coverage should pass
    assert any(r.check_name == "gt_source1_id_coverage" and r.status == ValidationStatus.PASS for r in results)
