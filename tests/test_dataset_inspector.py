"""Unit tests for dataset inspection module."""

from pathlib import Path
from dataclasses import asdict
import pytest

from business_entity_resolution.dataset_inspector import (
    inspect_file,
    FileInspectionReport,
)


def test_inspect_file(tmp_path):
    """Test detailed file inspection on sample TSV fixture."""
    test_file = tmp_path / "train_source1.tsv"
    content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAlpha Inc\t100 Main St\tUS\n"
        "S1-002\tBeta Solutions\t\tIndia\n"
    )
    test_file.write_text(content, encoding="utf-8")

    report = inspect_file(test_file, dataset_root=tmp_path)
    assert isinstance(report, FileInspectionReport)
    assert report.file_name == "train_source1.tsv"
    assert report.row_count == 2
    assert report.col_count == 4
    assert report.delimiter == "\t"
    assert report.split == "train"

    # Check column details
    addr_col = report.column_details["business_address"]
    assert addr_col.empty_str_count == 1
    assert addr_col.missing_count == 0

    name_col = report.column_details["business_name"]
    assert name_col.empty_str_count == 0
    assert name_col.unique_count == 2
    assert name_col.min_length == len("Alpha Inc")
    assert name_col.max_length == len("Beta Solutions")

    # Check JSON exportability
    rep_dict = asdict(report)
    assert rep_dict["row_count"] == 2
    assert "entity_id" in rep_dict["column_details"]
