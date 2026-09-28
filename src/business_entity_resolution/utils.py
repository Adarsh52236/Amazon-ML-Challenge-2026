"""Utility functions for entity resolution data processing, string parsing,
formatting, and file operations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_entity_id(entity_id: str) -> Tuple[str, str]:
    """Parse an entity identifier into its source prefix and identifier body.

    Examples
    --------
    >>> parse_entity_id("S1-925783039")
    ('S1-', '925783039')
    """
    if not isinstance(entity_id, str):
        raise TypeError(f"Entity ID must be a string, got {type(entity_id)}")
    eid = entity_id.strip()
    if "-" in eid:
        parts = eid.split("-", 1)
        return f"{parts[0]}-", parts[1]
    return "", eid


def validate_entity_id(
    entity_id: str, expected_prefix: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Validate that an entity ID is non-empty, clean, and follows prefix rules.

    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message_if_invalid)
    """
    if not isinstance(entity_id, str):
        return False, f"ID is not a string (type: {type(entity_id).__name__})"
    if not entity_id:
        return False, "ID is empty string"
    if entity_id.strip() != entity_id:
        return False, "ID contains leading or trailing whitespace"
    if expected_prefix and not entity_id.startswith(expected_prefix):
        return False, f"ID does not start with expected prefix '{expected_prefix}'"
    return True, None


def parse_matched_ids(matched_str: Optional[str]) -> List[str]:
    """Parse a comma-separated list of entity IDs from ground truth or prediction.

    Handles empty values, singletons, whitespace, and preserves string type.

    Parameters
    ----------
    matched_str : Optional[str]
        Raw string containing comma-separated IDs (or None / empty string).

    Returns
    -------
    List[str]
        Clean list of entity ID strings.
    """
    if matched_str is None:
        return []
    s = str(matched_str).strip()
    if not s or s.lower() == "nan":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def format_matched_ids(matched_ids: Iterable[str]) -> str:
    """Format an iterable of matched entity IDs into a standardized comma-separated string.

    Ensures no whitespace around commas, deduplicates preserving order, and handles empty sets.
    """
    seen = set()
    ordered_unique: List[str] = []
    for mid in matched_ids:
        mid_clean = str(mid).strip()
        if mid_clean and mid_clean not in seen:
            seen.add(mid_clean)
            ordered_unique.append(mid_clean)
    return ",".join(ordered_unique)


def format_bytes(size_bytes: int) -> str:
    """Convert byte count to human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def safe_read_header(
    file_path: Path, delimiter: str = "\t", encoding: str = "utf-8"
) -> List[str]:
    """Read just the column headers from a delimited file without loading data."""
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, [])
        return [col.strip() for col in header]


def count_file_lines(
    file_path: Path, encoding: str = "utf-8"
) -> Tuple[int, int]:
    """Count total physical lines and data lines (excluding 1 header row) in a file.

    Memory efficient streaming counter.
    """
    total_lines = 0
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for _ in f:
            total_lines += 1
    data_lines = max(0, total_lines - 1)
    return total_lines, data_lines


def build_directory_tree(
    root_path: Path, max_depth: int = 3, current_depth: int = 0
) -> str:
    """Generate a clean visual ASCII directory tree representation."""
    if not root_path.exists():
        return f"Directory does not exist: {root_path}"

    lines = []
    if current_depth == 0:
        lines.append(f"{root_path.name}/")

    if current_depth >= max_depth:
        return "\n".join(lines)

    try:
        entries = sorted(
            [e for e in root_path.iterdir() if not e.name.startswith(".")],
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except PermissionError:
        return "\n".join(lines)

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        prefix = "    " * current_depth

        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            sub_tree = build_directory_tree(entry, max_depth, current_depth + 1)
            # Add sub-lines if any were generated (skip root line of sub-tree)
            sub_lines = sub_tree.splitlines()
            if len(sub_lines) > 1:
                lines.extend(sub_lines[1:])
        else:
            size_str = format_bytes(entry.stat().st_size)
            lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

    return "\n".join(lines)
