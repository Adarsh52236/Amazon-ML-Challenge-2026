"""Data loading module for Business Entity Resolution.

Provides memory-efficient, type-safe, and robust utilities to load tabular
source data and ground-truth labels while strictly preserving string IDs and raw values.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

from business_entity_resolution.config import (
    get_data_base_dir,
    get_dataset_root,
    get_ground_truth_path,
    get_source_path,
    get_test_dir,
    get_train_dir,
)
from business_entity_resolution.utils import format_bytes, parse_matched_ids

logger = logging.getLogger(__name__)

# Canonical schema definitions
EXPECTED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]
EXPECTED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def discover_dataset_files(
    root_dir: Optional[Union[Path, str]] = None
) -> Dict[str, Path]:
    """Discover all canonical dataset files in the dataset folder.

    Parameters
    ----------
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path override.

    Returns
    -------
    Dict[str, Path]
        Mapping of semantic file keys to resolved Path objects.
    """
    base_dir = get_data_base_dir(root_dir)
    discovered: Dict[str, Path] = {}

    # Check training files
    train_dir = base_dir / "train"
    if train_dir.is_dir():
        for s_idx in (1, 2, 3):
            p = train_dir / f"train_source{s_idx}.tsv"
            if p.is_file():
                discovered[f"train_source{s_idx}"] = p
        gt_p = train_dir / "train_ground_truth.tsv"
        if gt_p.is_file():
            discovered["train_ground_truth"] = gt_p

    # Check test files
    test_dir = base_dir / "test"
    if test_dir.is_dir():
        for s_idx in (1, 2, 3):
            p = test_dir / f"test_source{s_idx}.tsv"
            if p.is_file():
                discovered[f"test_source{s_idx}"] = p

    return discovered


def load_tabular_file(
    file_path: Union[Path, str],
    sep: Optional[str] = None,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    encoding: str = "utf-8",
    chunksize: Optional[int] = None,
    keep_default_na: bool = False,
    na_values: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load a delimited tabular file (TSV or CSV) into a pandas DataFrame.

    Ensures:
    - String data types are preserved across all columns (preventing numeric casting of IDs).
    - Missing values are loaded faithfully without silently altering data.
    - Explicit encoding and separator handling.

    Parameters
    ----------
    file_path : Union[Path, str]
        Path to the file to load.
    sep : Optional[str]
        Field delimiter. If None, inferred from extension (.tsv -> '\t', .csv -> ',').
    nrows : Optional[int]
        Number of rows to read.
    usecols : Optional[List[str]]
        Specific columns to read.
    encoding : str
        File character encoding (default: 'utf-8').
    chunksize : Optional[int]
        If specified, returns an iterator of DataFrames.
    keep_default_na : bool
        Whether to convert standard NA values (NA, NaN, etc.) to float NaN.
        Default is False so literal text like 'NA' (e.g. country codes) isn't mangled.
    na_values : Optional[List[str]]
        Explicit strings to treat as NaN.

    Returns
    -------
    Union[pd.DataFrame, Iterator[pd.DataFrame]]
        Loaded DataFrame or chunk iterator.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tabular file not found: '{path}'")

    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","

    logger.info(
        f"Loading tabular file: '{path.name}' ({format_bytes(path.stat().st_size)}) with sep={repr(sep)}"
    )

    # Use dtype=str to strictly preserve leading zeros and formatting in IDs
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        nrows=nrows,
        usecols=usecols,
        encoding=encoding,
        chunksize=chunksize,
        keep_default_na=keep_default_na,
        na_values=na_values,
        engine="c",
    )


def load_source_file(
    split: str,
    source_num: int,
    root_dir: Optional[Union[Path, str]] = None,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load an official source TSV file (Source 1, 2, or 3) for train or test split.

    Parameters
    ----------
    split : str
        'train' or 'test'.
    source_num : int
        1, 2, or 3.
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path override.
    nrows : Optional[int]
        Number of rows to load.
    usecols : Optional[List[str]]
        Columns to load.
    chunksize : Optional[int]
        If provided, returns chunk iterator.

    Returns
    -------
    Union[pd.DataFrame, Iterator[pd.DataFrame]]
        DataFrame containing entity records with columns:
        ['entity_id', 'business_name', 'business_address', 'country'].
    """
    file_path = get_source_path(split, source_num, root_dir)
    return load_tabular_file(
        file_path,
        sep="\t",
        nrows=nrows,
        usecols=usecols,
        chunksize=chunksize,
    )


def load_ground_truth(
    root_dir: Optional[Union[Path, str]] = None,
    nrows: Optional[int] = None,
    as_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, List[str]]]:
    """Load official training ground truth file.

    Parameters
    ----------
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path override.
    nrows : Optional[int]
        Number of rows to read.
    as_dict : bool
        If True, returns a dictionary mapping `source1_entity_id -> list of matched_ids`.
        If False, returns raw DataFrame with columns:
        ['source1_entity_id', 'matched_entity_ids'].

    Returns
    -------
    Union[pd.DataFrame, Dict[str, List[str]]]
        Ground truth DataFrame or dictionary mapping.
    """
    file_path = get_ground_truth_path(root_dir)
    df = load_tabular_file(file_path, sep="\t", nrows=nrows)

    if as_dict:
        mapping: Dict[str, List[str]] = {}
        for s1_id, matched_str in zip(
            df["source1_entity_id"], df["matched_entity_ids"]
        ):
            mapping[s1_id] = parse_matched_ids(matched_str)
        return mapping

    return df


def iter_source_records(
    file_path: Union[Path, str],
    encoding: str = "utf-8",
) -> Iterator[Dict[str, str]]:
    """Stream records from a source TSV line-by-line as dictionaries.

    Extremely memory efficient for large 5M+ row files.
    """
    path = Path(file_path).resolve()
    with open(path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def load_training_data(
    root_dir: Optional[Union[Path, str]] = None,
    sources: Optional[List[int]] = None,
    load_gt: bool = True,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """Load training sources and optionally ground truth.

    Parameters
    ----------
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path.
    sources : Optional[List[int]]
        List of source numbers to load (e.g. [1, 2, 3]). Defaults to [1].
    load_gt : bool
        Whether to load ground truth (default: True).
    nrows : Optional[int]
        Max rows per file (useful for fast experimentation).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'source1', 'source2', 'source3', 'ground_truth'.
    """
    if sources is None:
        sources = [1]

    data: Dict[str, Any] = {}
    for s_num in sources:
        data[f"source{s_num}"] = load_source_file(
            split="train",
            source_num=s_num,
            root_dir=root_dir,
            nrows=nrows,
        )

    if load_gt:
        data["ground_truth"] = load_ground_truth(root_dir=root_dir, nrows=nrows)

    return data


def load_test_data(
    root_dir: Optional[Union[Path, str]] = None,
    sources: Optional[List[int]] = None,
    nrows: Optional[int] = None,
) -> Dict[str, Any]:
    """Load test sources.

    Parameters
    ----------
    root_dir : Optional[Union[Path, str]]
        Optional dataset root path.
    sources : Optional[List[int]]
        List of source numbers to load (defaults to [1]).
    nrows : Optional[int]
        Max rows per file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys 'source1', 'source2', 'source3'.
    """
    if sources is None:
        sources = [1]

    data: Dict[str, Any] = {}
    for s_num in sources:
        data[f"source{s_num}"] = load_source_file(
            split="test",
            source_num=s_num,
            root_dir=root_dir,
            nrows=nrows,
        )

    return data
