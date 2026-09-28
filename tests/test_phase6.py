"""
Phase 6 production submission validation tests.

Tests cover:
- Submission file schema and row count
- Entity ID format validation
- Candidate/match subset relationship
- Multi-match preservation (no argmax enforcement)
- Production model loadability
- Feature schema consistency
- Pipeline stats JSON integrity
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DATA_DIR = ROOT / "student_resource" / "dataset" / "test"
MODEL_DIR = ROOT / "models"

S1_EXPECTED_ROWS = 1_732_544


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_submission(path: Path) -> pd.DataFrame:
    """Load and minimally validate a submission TSV."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def get_id_set(comma_str: str) -> set:
    if not comma_str or comma_str.strip() == "":
        return set()
    return set(comma_str.strip().split(","))


# ─── Model file tests ─────────────────────────────────────────────────────────

def test_model_xgb_file_exists():
    p = MODEL_DIR / "model_xgb_prod.json"
    assert p.exists(), f"XGBoost model not found: {p}"
    assert p.stat().st_size > 100_000, "XGBoost model too small (<100 KB)"


def test_model_lgb_file_exists():
    p = MODEL_DIR / "model_lgb_prod.txt"
    assert p.exists(), f"LightGBM model not found: {p}"
    assert p.stat().st_size > 50_000, "LightGBM model too small (<50 KB)"


def test_model_xgb_loadable():
    import xgboost as xgb
    m = xgb.Booster()
    m.load_model(str(MODEL_DIR / "model_xgb_prod.json"))
    feature_names = m.feature_names
    assert feature_names is not None
    assert len(feature_names) == 54, f"Expected 54 features, got {len(feature_names)}"


def test_model_lgb_loadable():
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(MODEL_DIR / "model_lgb_prod.txt"))
    n_feat = m.num_feature()
    assert n_feat == 54, f"Expected 54 features, got {n_feat}"


# ─── Feature schema consistency ───────────────────────────────────────────────

def test_feature_schema_consistent_xgb_lgb():
    """XGBoost and LightGBM must agree on feature count."""
    import xgboost as xgb
    import lightgbm as lgb
    xgb_m = xgb.Booster()
    xgb_m.load_model(str(MODEL_DIR / "model_xgb_prod.json"))
    lgb_m = lgb.Booster(model_file=str(MODEL_DIR / "model_lgb_prod.txt"))
    assert len(xgb_m.feature_names) == lgb_m.num_feature()


def test_feature_schema_matches_extractor():
    """PairwiseFeatureExtractor must produce exactly 54 features."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
    extractor = PairwiseFeatureExtractor()
    assert len(extractor.feature_names) == 54


# ─── Submission file schema tests ─────────────────────────────────────────────

@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_matching_results_header():
    df = load_submission(OUTPUT / "matching_results.tsv")
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"], \
        f"Unexpected columns: {list(df.columns)}"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_matching_results_row_count():
    df = load_submission(OUTPUT / "matching_results.tsv")
    assert len(df) == S1_EXPECTED_ROWS, \
        f"Expected {S1_EXPECTED_ROWS:,} rows, got {len(df):,}"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_matching_results_s1_ids_format():
    df = load_submission(OUTPUT / "matching_results.tsv")
    bad = df[~df["source1_entity_id"].str.startswith("S1-")]
    assert len(bad) == 0, f"Found {len(bad):,} rows with non-S1- entity IDs"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_matching_results_target_ids_format():
    """All matched_entity_ids must be S2- or S3- prefixed (or empty)."""
    df = load_submission(OUTPUT / "matching_results.tsv")
    n_nonempty = 0
    n_bad = 0
    for row in df["matched_entity_ids"].iloc[:50000]:  # sample first 50k for speed
        if not row or row.strip() == "":
            continue
        n_nonempty += 1
        for tid in row.split(","):
            tid = tid.strip()
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                n_bad += 1
    assert n_bad == 0, f"Found {n_bad:,} target IDs not starting with S2- or S3-"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_matching_results_no_duplicate_s1_ids():
    df = load_submission(OUTPUT / "matching_results.tsv")
    dupes = df["source1_entity_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes:,} duplicate source1_entity_id rows"


@pytest.mark.skipif(
    not (OUTPUT / "candidate_pairs.tsv").exists(),
    reason="candidate_pairs.tsv not yet generated"
)
def test_candidate_pairs_header():
    df = load_submission(OUTPUT / "candidate_pairs.tsv")
    assert list(df.columns) == ["source1_entity_id", "candidate_entity_ids"], \
        f"Unexpected columns: {list(df.columns)}"


@pytest.mark.skipif(
    not (OUTPUT / "candidate_pairs.tsv").exists(),
    reason="candidate_pairs.tsv not yet generated"
)
def test_candidate_pairs_row_count():
    df = load_submission(OUTPUT / "candidate_pairs.tsv")
    assert len(df) == S1_EXPECTED_ROWS, \
        f"Expected {S1_EXPECTED_ROWS:,} rows, got {len(df):,}"


@pytest.mark.skipif(
    not (OUTPUT / "candidate_pairs.tsv").exists() or
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="Submission files not yet generated"
)
def test_matches_subset_of_candidates():
    """Every match must appear as a candidate for that S1 entity (sample check)."""
    cands_df = load_submission(OUTPUT / "candidate_pairs.tsv")
    match_df = load_submission(OUTPUT / "matching_results.tsv")

    # Sample 1000 rows for speed
    sample_idx = np.random.default_rng(42).choice(len(match_df), min(1000, len(match_df)), replace=False)

    violations = 0
    for i in sample_idx:
        s1_id = match_df.iloc[i]["source1_entity_id"]
        matches = get_id_set(match_df.iloc[i]["matched_entity_ids"])
        cands_row = cands_df[cands_df["source1_entity_id"] == s1_id]
        if cands_row.empty:
            if matches:
                violations += 1
            continue
        cands = get_id_set(cands_row.iloc[0]["candidate_entity_ids"])
        extra = matches - cands
        if extra:
            violations += 1

    assert violations == 0, \
        f"Found {violations} S1 entities with matches outside their candidate set"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_multi_match_preserved():
    """Verify that some S1 entities have >1 match (multi-match is preserved)."""
    df = load_submission(OUTPUT / "matching_results.tsv")
    nonempty = df[df["matched_entity_ids"] != ""]
    if len(nonempty) == 0:
        pytest.skip("No matches found — cannot check multi-match")
    multi = nonempty[nonempty["matched_entity_ids"].str.contains(",")]
    assert len(multi) > 0, \
        "No multi-match rows found — argmax constraint may have been incorrectly applied"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists(),
    reason="matching_results.tsv not yet generated"
)
def test_s1_order_matches_source_file():
    """Row order in matching_results.tsv must match test_source1.tsv order."""
    s1_df = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str)
    match_df = load_submission(OUTPUT / "matching_results.tsv")
    assert len(match_df) == len(s1_df)
    assert list(match_df["source1_entity_id"]) == list(s1_df["entity_id"]), \
        "Row order in matching_results.tsv does not match test_source1.tsv"


@pytest.mark.skipif(
    not (OUTPUT / "matching_results_xgb.tsv").exists(),
    reason="XGB submission not yet generated"
)
def test_xgb_lgb_blend_row_counts_equal():
    """All 4 output files must have identical row counts."""
    files = [
        "matching_results_xgb.tsv",
        "matching_results_lgb.tsv",
        "matching_results_blend.tsv",
        "matching_results.tsv",
    ]
    counts = {}
    for fname in files:
        p = OUTPUT / fname
        if p.exists():
            counts[fname] = sum(1 for _ in open(p, encoding="utf-8")) - 1  # minus header
    if len(counts) < 2:
        pytest.skip("Not all submission files generated yet")
    unique_counts = set(counts.values())
    assert len(unique_counts) == 1, \
        f"Mismatched row counts: {counts}"


# ─── Pipeline stats JSON tests ────────────────────────────────────────────────

@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_keys():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    required_keys = [
        "total_s1", "total_targets", "total_candidates",
        "total_matches_xgb", "total_matches_lgb", "total_matches_blend",
        "elapsed_total_s", "country_stats", "checksums", "thresholds",
    ]
    for k in required_keys:
        assert k in stats, f"Missing key '{k}' in pipeline stats JSON"


@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_s1_count():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    assert stats["total_s1"] == S1_EXPECTED_ROWS, \
        f"Expected {S1_EXPECTED_ROWS:,} S1, got {stats['total_s1']:,}"


@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_all_countries_present():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    country_stats = stats.get("country_stats", {})
    for country in ["France", "US", "India"]:
        assert country in country_stats, f"Missing country '{country}' in stats"


@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_thresholds():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    th = stats["thresholds"]
    assert th["xgb"] == pytest.approx(0.45)
    assert th["lgb"] == pytest.approx(0.30)
    assert th["blend"] == pytest.approx(0.45)


@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_checksums_present():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    checksums = stats.get("checksums", {})
    expected = ["candidates", "xgb", "lgb", "blend", "champion"]
    for k in expected:
        assert k in checksums, f"Missing checksum for '{k}'"
        # SHA-256 is 64 hex chars
        assert len(checksums[k]) == 64, f"Invalid SHA-256 for '{k}': {checksums[k]}"


@pytest.mark.skipif(
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Pipeline stats JSON not yet generated"
)
def test_pipeline_stats_xgb_matches_positive():
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    assert stats["total_matches_xgb"] > 0, "XGBoost produced zero matches — check threshold/models"
    assert stats["total_matches_lgb"] > 0, "LightGBM produced zero matches — check threshold/models"


# ─── Submission file integrity (checksum stability) ───────────────────────────

@pytest.mark.skipif(
    not (OUTPUT / "matching_results.tsv").exists() or
    not (OUTPUT / "phase_6_pipeline_stats.json").exists(),
    reason="Submission files not yet generated"
)
def test_champion_checksum_stable():
    """SHA-256 of matching_results.tsv must match value recorded in pipeline stats."""
    import hashlib
    with open(OUTPUT / "phase_6_pipeline_stats.json") as f:
        stats = json.load(f)
    recorded_sha = stats["checksums"].get("champion", "")
    if not recorded_sha:
        pytest.skip("No champion checksum in stats")
    h = hashlib.sha256()
    with open(OUTPUT / "matching_results.tsv", "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    assert actual_sha == recorded_sha, \
        f"matching_results.tsv checksum mismatch!\n  recorded: {recorded_sha}\n  actual:   {actual_sha}"
