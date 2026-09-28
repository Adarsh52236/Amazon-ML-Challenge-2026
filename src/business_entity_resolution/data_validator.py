"""Data validation module for Business Entity Resolution.

Performs rigorous, non-destructive validation of official competition files,
schemas, ID integrity, cross-table ground truth references, and data quality.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from business_entity_resolution.config import (
    configure_console_encoding,
    get_dataset_root,
)
from business_entity_resolution.data_loader import (
    EXPECTED_GROUND_TRUTH_COLUMNS,
    EXPECTED_SOURCE_COLUMNS,
    discover_dataset_files,
)
from business_entity_resolution.utils import format_bytes, parse_matched_ids

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    check_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    affected_rows: Optional[int] = None

    def __str__(self) -> str:
        loc = f" [{Path(self.file_path).name}]" if self.file_path else ""
        aff = f" (affected rows: {self.affected_rows:,})" if self.affected_rows is not None else ""
        return f"[{self.status.value}] {self.check_name}{loc}: {self.message}{aff}"


def check_file_existence(file_path: Path, required: bool = True) -> ValidationResult:
    """Validate that a required file exists, is non-empty, and readable."""
    if not file_path.exists():
        return ValidationResult(
            check_name="file_existence",
            status=ValidationStatus.ERROR if required else ValidationStatus.WARNING,
            severity=ValidationSeverity.CRITICAL if required else ValidationSeverity.MEDIUM,
            message=f"File does not exist at '{file_path}'",
            file_path=str(file_path),
        )

    if not file_path.is_file():
        return ValidationResult(
            check_name="file_existence",
            status=ValidationStatus.ERROR,
            severity=ValidationSeverity.CRITICAL,
            message=f"Path is not a regular file: '{file_path}'",
            file_path=str(file_path),
        )

    size = file_path.stat().st_size
    if size == 0:
        return ValidationResult(
            check_name="file_empty_check",
            status=ValidationStatus.ERROR,
            severity=ValidationSeverity.CRITICAL,
            message=f"File is completely empty (0 bytes): '{file_path}'",
            file_path=str(file_path),
        )

    return ValidationResult(
        check_name="file_existence",
        status=ValidationStatus.PASS,
        severity=ValidationSeverity.INFO,
        message=f"File verified ({format_bytes(size)})",
        file_path=str(file_path),
    )


def check_tabular_schema(
    file_path: Path,
    expected_columns: List[str],
    delimiter: str = "\t",
) -> Tuple[List[ValidationResult], List[str]]:
    """Validate header row, columns presence, and detect unexpected/duplicate columns."""
    results: List[ValidationResult] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
    except Exception as exc:
        results.append(
            ValidationResult(
                check_name="schema_read_header",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to read header from '{file_path}': {exc}",
                file_path=str(file_path),
            )
        )
        return results, []

    if header is None:
        results.append(
            ValidationResult(
                check_name="schema_empty_header",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"File has no header row: '{file_path}'",
                file_path=str(file_path),
            )
        )
        return results, []

    clean_header = [c.strip() for c in header]

    # Check duplicates in header
    if len(clean_header) != len(set(clean_header)):
        results.append(
            ValidationResult(
                check_name="schema_duplicate_columns",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.HIGH,
                message=f"Header contains duplicate column names: {clean_header}",
                file_path=str(file_path),
            )
        )

    # Check missing required columns
    missing_cols = set(expected_columns) - set(clean_header)
    if missing_cols:
        results.append(
            ValidationResult(
                check_name="schema_missing_columns",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required columns {sorted(missing_cols)}. Found: {clean_header}",
                file_path=str(file_path),
            )
        )

    # Check unexpected columns
    unexpected_cols = set(clean_header) - set(expected_columns)
    if unexpected_cols:
        results.append(
            ValidationResult(
                check_name="schema_unexpected_columns",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.LOW,
                message=f"File contains additional non-standard columns: {sorted(unexpected_cols)}",
                file_path=str(file_path),
            )
        )

    if not missing_cols and not unexpected_cols:
        results.append(
            ValidationResult(
                check_name="schema_validation",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message=f"Exact expected schema verified: {clean_header}",
                file_path=str(file_path),
            )
        )

    return results, clean_header


def validate_source_file_content(
    file_path: Path,
    expected_prefix: str,
    max_scan_rows: Optional[int] = None,
) -> Tuple[List[ValidationResult], Set[str]]:
    """Deep validation of source file rows: null IDs, empty strings, duplicates, prefixes, and addresses.

    Returns list of validation results and the set of entity IDs.
    """
    results: List[ValidationResult] = []
    seen_ids: Set[str] = set()
    dup_ids = 0
    empty_ids = 0
    whitespace_ids = 0
    prefix_mismatches = 0
    empty_names = 0
    empty_addresses = 0
    empty_countries = 0
    malformed_rows = 0
    total_rows = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return results, seen_ids

        for row in reader:
            total_rows += 1
            if max_scan_rows and total_rows > max_scan_rows:
                break

            if len(row) < len(header):
                malformed_rows += 1
                continue

            eid = row[0]
            name = row[1] if len(row) > 1 else ""
            addr = row[2] if len(row) > 2 else ""
            ctry = row[3] if len(row) > 3 else ""

            # ID validations
            if not eid:
                empty_ids += 1
            elif eid.strip() == "":
                whitespace_ids += 1
            else:
                if not eid.startswith(expected_prefix):
                    prefix_mismatches += 1
                if eid in seen_ids:
                    dup_ids += 1
                else:
                    seen_ids.add(eid)

            # Business field validations
            if not name or name.strip() == "":
                empty_names += 1
            if not addr or addr.strip() == "":
                empty_addresses += 1
            if not ctry or ctry.strip() == "":
                empty_countries += 1

    # Evaluate results
    if malformed_rows > 0:
        results.append(
            ValidationResult(
                check_name="malformed_row_check",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {malformed_rows:,} malformed rows with fewer columns than header",
                file_path=str(file_path),
                affected_rows=malformed_rows,
            )
        )

    if empty_ids > 0 or whitespace_ids > 0:
        results.append(
            ValidationResult(
                check_name="entity_id_empty_check",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found empty or whitespace entity IDs (empty: {empty_ids}, whitespace: {whitespace_ids})",
                file_path=str(file_path),
                affected_rows=empty_ids + whitespace_ids,
            )
        )

    if dup_ids > 0:
        results.append(
            ValidationResult(
                check_name="entity_id_uniqueness",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {dup_ids:,} duplicate entity IDs",
                file_path=str(file_path),
                affected_rows=dup_ids,
            )
        )
    else:
        results.append(
            ValidationResult(
                check_name="entity_id_uniqueness",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message=f"All {len(seen_ids):,} entity IDs are 100% unique",
                file_path=str(file_path),
            )
        )

    if prefix_mismatches > 0:
        results.append(
            ValidationResult(
                check_name="entity_id_prefix_format",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.HIGH,
                message=f"Found {prefix_mismatches:,} IDs not starting with expected prefix '{expected_prefix}'",
                file_path=str(file_path),
                affected_rows=prefix_mismatches,
            )
        )
    else:
        results.append(
            ValidationResult(
                check_name="entity_id_prefix_format",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message=f"All IDs strictly adhere to '{expected_prefix}' prefix",
                file_path=str(file_path),
            )
        )

    if empty_names > 0:
        results.append(
            ValidationResult(
                check_name="business_name_missing",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.MEDIUM,
                message=f"Found {empty_names:,} missing or empty business names",
                file_path=str(file_path),
                affected_rows=empty_names,
            )
        )

    if empty_addresses > 0:
        pct = (empty_addresses / max(1, total_rows)) * 100.0
        results.append(
            ValidationResult(
                check_name="business_address_missing",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.LOW,
                message=f"Found {empty_addresses:,} empty addresses ({pct:.2f}% of source)",
                file_path=str(file_path),
                affected_rows=empty_addresses,
            )
        )

    if empty_countries > 0:
        results.append(
            ValidationResult(
                check_name="country_missing",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.HIGH,
                message=f"Found {empty_countries:,} records with missing country",
                file_path=str(file_path),
                affected_rows=empty_countries,
            )
        )

    return results, seen_ids


def validate_ground_truth_file(
    gt_path: Path,
    s1_ids: Set[str],
    s2_ids: Optional[Set[str]] = None,
    s3_ids: Optional[Set[str]] = None,
) -> List[ValidationResult]:
    """Validate ground truth formatting, reference integrity, and singletons."""
    results: List[ValidationResult] = []

    total_rows = 0
    gt_s1_ids: Set[str] = set()
    dup_s1_ids = 0
    singletons = 0
    total_matches = 0
    intra_row_dups = 0
    foreign_s1_ids = 0
    missing_s2_refs = 0
    missing_s3_refs = 0
    invalid_prefixes = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            total_rows += 1
            if len(row) < 1:
                continue
            s1_id = row[0].strip()
            if s1_id in gt_s1_ids:
                dup_s1_ids += 1
            else:
                gt_s1_ids.add(s1_id)

            if s1_ids and s1_id not in s1_ids:
                foreign_s1_ids += 1

            matched_str = row[1].strip() if len(row) > 1 else ""
            if not matched_str:
                singletons += 1
            else:
                matches = parse_matched_ids(matched_str)
                total_matches += len(matches)
                if len(matches) != len(set(matches)):
                    intra_row_dups += 1

                for mid in matches:
                    if mid.startswith("S2-"):
                        if s2_ids is not None and mid not in s2_ids:
                            missing_s2_refs += 1
                    elif mid.startswith("S3-"):
                        if s3_ids is not None and mid not in s3_ids:
                            missing_s3_refs += 1
                    else:
                        invalid_prefixes += 1

    # Evaluate checks
    if dup_s1_ids > 0:
        results.append(
            ValidationResult(
                check_name="gt_source1_id_uniqueness",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Ground truth has {dup_s1_ids:,} duplicate source1_entity_id rows",
                file_path=str(gt_path),
                affected_rows=dup_s1_ids,
            )
        )
    else:
        results.append(
            ValidationResult(
                check_name="gt_source1_id_uniqueness",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message=f"All {len(gt_s1_ids):,} ground truth S1 rows have unique IDs",
                file_path=str(gt_path),
            )
        )

    if foreign_s1_ids > 0:
        results.append(
            ValidationResult(
                check_name="gt_source1_id_cross_ref",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {foreign_s1_ids:,} IDs in ground truth that do not exist in train_source1.tsv",
                file_path=str(gt_path),
                affected_rows=foreign_s1_ids,
            )
        )
    elif s1_ids:
        missing_s1_in_gt = len(s1_ids - gt_s1_ids)
        if missing_s1_in_gt > 0:
            results.append(
                ValidationResult(
                    check_name="gt_source1_id_coverage",
                    status=ValidationStatus.ERROR,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"{missing_s1_in_gt:,} Source 1 IDs from train_source1.tsv are missing from ground truth",
                    file_path=str(gt_path),
                    affected_rows=missing_s1_in_gt,
                )
            )
        else:
            results.append(
                ValidationResult(
                    check_name="gt_source1_id_coverage",
                    status=ValidationStatus.PASS,
                    severity=ValidationSeverity.INFO,
                    message="Ground truth matches train_source1.tsv entities exactly 1-to-1 (100% coverage)",
                    file_path=str(gt_path),
                )
            )

    if intra_row_dups > 0:
        results.append(
            ValidationResult(
                check_name="gt_intra_row_duplicate_matches",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.HIGH,
                message=f"Found {intra_row_dups:,} rows with duplicate matched IDs in their match list",
                file_path=str(gt_path),
                affected_rows=intra_row_dups,
            )
        )
    else:
        results.append(
            ValidationResult(
                check_name="gt_intra_row_duplicate_matches",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message="Zero duplicate matched IDs within any ground truth match list",
                file_path=str(gt_path),
            )
        )

    if invalid_prefixes > 0:
        results.append(
            ValidationResult(
                check_name="gt_target_prefixes",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.HIGH,
                message=f"Found {invalid_prefixes:,} matched IDs with invalid prefixes (must be S2- or S3-)",
                file_path=str(gt_path),
                affected_rows=invalid_prefixes,
            )
        )

    if s2_ids is not None and missing_s2_refs > 0:
        results.append(
            ValidationResult(
                check_name="gt_s2_reference_integrity",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Ground truth references {missing_s2_refs:,} S2 IDs that do not exist in train_source2.tsv",
                file_path=str(gt_path),
                affected_rows=missing_s2_refs,
            )
        )
    elif s2_ids is not None:
        results.append(
            ValidationResult(
                check_name="gt_s2_reference_integrity",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message="All referenced S2 IDs exist in train_source2.tsv",
                file_path=str(gt_path),
            )
        )

    if s3_ids is not None and missing_s3_refs > 0:
        results.append(
            ValidationResult(
                check_name="gt_s3_reference_integrity",
                status=ValidationStatus.ERROR,
                severity=ValidationSeverity.CRITICAL,
                message=f"Ground truth references {missing_s3_refs:,} S3 IDs that do not exist in train_source3.tsv",
                file_path=str(gt_path),
                affected_rows=missing_s3_refs,
            )
        )
    elif s3_ids is not None:
        results.append(
            ValidationResult(
                check_name="gt_s3_reference_integrity",
                status=ValidationStatus.PASS,
                severity=ValidationSeverity.INFO,
                message="All referenced S3 IDs exist in train_source3.tsv",
                file_path=str(gt_path),
            )
        )

    # Information on singletons
    results.append(
        ValidationResult(
            check_name="gt_singletons_distribution",
            status=ValidationStatus.PASS,
            severity=ValidationSeverity.INFO,
            message=f"Singletons (0 matches): {singletons:,} ({singletons/max(1,total_rows)*100:.2f}%); Entities with matches: {total_rows - singletons:,}",
            file_path=str(gt_path),
        )
    )

    return results


def validate_dataset(
    root_dir: Optional[Union[Path, str]] = None,
    verify_gt_cross_refs: bool = True,
    sample_scan_limit: Optional[int] = None,
) -> List[ValidationResult]:
    """Run complete validation suite across all official dataset files.

    Parameters
    ----------
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path.
    verify_gt_cross_refs : bool
        Whether to perform cross-table reference checking between ground truth and source files.
    sample_scan_limit : Optional[int]
        If set, limits row scanning for fast unit testing.
    """
    configure_console_encoding()
    results: List[ValidationResult] = []
    root = get_dataset_root(root_dir)

    discovered = discover_dataset_files(root)
    required_keys = [
        "train_source1",
        "train_source2",
        "train_source3",
        "train_ground_truth",
        "test_source1",
        "test_source2",
        "test_source3",
    ]

    for req in required_keys:
        if req not in discovered:
            results.append(
                ValidationResult(
                    check_name="required_file_discovery",
                    status=ValidationStatus.ERROR,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Required dataset component '{req}' could not be discovered.",
                )
            )
        else:
            file_path = discovered[req]
            results.append(check_file_existence(file_path, required=True))

            # Validate schema
            expected_cols = (
                EXPECTED_GROUND_TRUTH_COLUMNS
                if req == "train_ground_truth"
                else EXPECTED_SOURCE_COLUMNS
            )
            schema_res, _ = check_tabular_schema(file_path, expected_cols)
            results.extend(schema_res)

    # Validate source content & accumulate IDs
    source_ids: Dict[str, Set[str]] = {}
    for key, p in discovered.items():
        if "source1" in key:
            prefix = "S1-"
        elif "source2" in key:
            prefix = "S2-"
        elif "source3" in key:
            prefix = "S3-"
        else:
            continue

        c_results, ids = validate_source_file_content(
            p, expected_prefix=prefix, max_scan_rows=sample_scan_limit
        )
        results.extend(c_results)
        source_ids[key] = ids

    # Validate ground truth if present
    if "train_ground_truth" in discovered:
        gt_path = discovered["train_ground_truth"]
        s1_ids = source_ids.get("train_source1", set())
        s2_ids = source_ids.get("train_source2") if verify_gt_cross_refs else None
        s3_ids = source_ids.get("train_source3") if verify_gt_cross_refs else None

        gt_results = validate_ground_truth_file(
            gt_path=gt_path,
            s1_ids=s1_ids,
            s2_ids=s2_ids,
            s3_ids=s3_ids,
        )
        results.extend(gt_results)

    return results


def summarize_validation(results: List[ValidationResult]) -> Tuple[int, int, int]:
    """Return counts of (pass_count, warning_count, error_count)."""
    passes = sum(1 for r in results if r.status == ValidationStatus.PASS)
    warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
    errors = sum(1 for r in results if r.status == ValidationStatus.ERROR)
    return passes, warnings, errors


def print_validation_report(results: List[ValidationResult]) -> None:
    """Print an attractive, categorized validation report to console."""
    configure_console_encoding()
    passes, warnings, errors = summarize_validation(results)

    print("=" * 70)
    print("        BUSINESS ENTITY RESOLUTION DATASET VALIDATION REPORT")
    print("=" * 70)

    # Print Errors
    error_items = [r for r in results if r.status == ValidationStatus.ERROR]
    if error_items:
        print(f"\n[!] ERRORS DETECTED ({len(error_items)}):")
        for item in error_items:
            print(f"  - {item}")
    else:
        print("\n[OK] Zero validation errors detected.")

    # Print Warnings
    warning_items = [r for r in results if r.status == ValidationStatus.WARNING]
    if warning_items:
        print(f"\n[*] WARNINGS ({len(warning_items)}):")
        for item in warning_items:
            print(f"  - {item}")
    else:
        print("\n[OK] Zero warnings detected.")

    # Summary
    print("\n" + "-" * 70)
    print(f"SUMMARY: {passes} PASSED | {warnings} WARNINGS | {errors} ERRORS")
    print("-" * 70)


def main_cli() -> None:
    """CLI entry point for dataset validation."""
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Validate official Business Entity Resolution datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to official dataset root directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors and exit with non-zero code.",
    )
    parser.add_argument(
        "--skip-id-cross-refs",
        action="store_true",
        help="Skip deep cross-table ID verification between ground truth and sources (faster).",
    )

    args = parser.parse_args()

    try:
        results = validate_dataset(
            root_dir=args.dataset_root,
            verify_gt_cross_refs=not args.skip_id_cross_refs,
        )
        print_validation_report(results)
        passes, warnings, errors = summarize_validation(results)

        if errors > 0:
            sys.exit(1)
        if args.strict and warnings > 0:
            sys.exit(2)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Validation crashed: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
