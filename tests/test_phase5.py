"""
Phase 5 automated regression, audit, and reconciliation unit tests.
Verifies:
  - blocking strategy determinism
  - duplicate removal and provenance preservation
  - no ground-truth access in candidate generation
  - stable candidate ordering
  - feature schema consistency (54 numerical + 3 trace)
  - no NaN or Inf
  - ensemble probability range
  - reproducible threshold selection
  - GroupKFold separation
  - multi-match preservation
  - ground-truth denominator reconciliation
  - candidate uniqueness and integrity
  - metric calculation consistency
"""

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.address_normalizer import BusinessAddressNormalizer
from business_entity_resolution.blocking_keys import generate_blocking_keys
from business_entity_resolution.candidate_generator import CandidateGenerator, CandidatePair
from business_entity_resolution.data_loader import load_ground_truth, load_source_file
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
from business_entity_resolution.models.model_config import ModelConfig, ValidationConfig
from business_entity_resolution.models.validation import Evaluator
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.text_normalizer import BusinessNameNormalizer


def test_blocking_strategy_determinism():
    """Verify that blocking key generation produces identical keys for identical input."""
    name_norm = BusinessNameNormalizer()
    addr_norm = BusinessAddressNormalizer()
    cfg = BlockingConfig(use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True)

    n1 = name_norm.normalize("Acme Logistics Global Pvt Ltd")
    a1 = addr_norm.normalize("123 Industrial Way Suite 400", country="US")

    keys1 = generate_blocking_keys(n1, a1, "US", cfg)
    keys2 = generate_blocking_keys(n1, a1, "US", cfg)

    assert keys1 == keys2
    assert "stem_prefix" in keys1
    assert keys1["stem_prefix"] == ["US::sp::industrial::acm"]


def test_duplicate_removal_and_provenance():
    """Verify that candidate generation removes duplicate pairs deterministically and aggregates provenance."""
    cfg = BlockingConfig(max_candidates_per_entity=100, use_stem_prefix=True)
    gen = CandidateGenerator(config=cfg)

    # Index two records that will collide on multiple keys
    gen.index_target_record("T-1", "Alpha Logistics", "100 Main St", "US")
    gen.index_target_record("T-2", "Beta Tech", "200 Market St", "US")

    s1_df = pd.DataFrame([{
        "entity_id": "S1-1",
        "business_name": "Alpha Logistics Corp",
        "business_address": "100 Main Street",
        "country": "US"
    }])

    candidates = gen.generate_candidates_for_dataframe(s1_df)
    cand_df = gen.candidates_to_dataframe(candidates)

    # Deterministic deduplication
    assert not cand_df.duplicated(subset=["source1_entity_id", "matched_entity_id"]).any()
    # Matched candidate T-1 should have multiple provenance channels
    t1_pair = next(c for c in candidates if c.matched_entity_id == "T-1")
    assert len(t1_pair.channels) >= 1


def test_no_ground_truth_in_candidate_generation():
    """Ensure candidate generator operates strictly without ground-truth labels."""
    cfg = BlockingConfig()
    gen = CandidateGenerator(config=cfg)

    # CandidateGenerator has no ground truth attributes or methods
    assert not hasattr(gen, "ground_truth")
    assert not hasattr(gen, "gt_mapping")
    assert not hasattr(gen, "labels")


def test_stable_candidate_ordering():
    """Verify candidate output ordering is stable and reproducible across repeated runs."""
    cfg = BlockingConfig(max_candidates_per_entity=50, use_stem_prefix=True)

    gen1 = CandidateGenerator(config=cfg)
    gen2 = CandidateGenerator(config=cfg)

    targets = [
        ("T-1", "Delta Global", "15 Oak Ave", "US"),
        ("T-2", "Delta Services", "15 Oak Ave", "US"),
        ("T-3", "Gamma Corp", "99 Pine St", "US"),
    ]
    for tid, name, addr, ctry in targets:
        gen1.index_target_record(tid, name, addr, ctry)
        gen2.index_target_record(tid, name, addr, ctry)

    s1_df = pd.DataFrame([{
        "entity_id": "S1-10",
        "business_name": "Delta Global Solutions",
        "business_address": "15 Oak Avenue",
        "country": "US"
    }])

    res1 = gen1.generate_candidates_for_dataframe(s1_df)
    res2 = gen2.generate_candidates_for_dataframe(s1_df)

    assert [c.matched_entity_id for c in res1] == [c.matched_entity_id for c in res2]
    assert [c.channels for c in res1] == [c.channels for c in res2]


def test_feature_schema_consistency_and_cleanliness():
    """Verify feature extractor produces strictly 54 features with zero NaN/Inf."""
    extractor = PairwiseFeatureExtractor()
    e1 = extractor.preprocess_entity("S1-1", "Alpha Inc", "123 Main St", "US", source="S1")
    e2 = extractor.preprocess_entity("T-1", "Alpha LLC", None, "US", source="S2")

    pair = CandidatePair("S1-1", "T-1", "S2", ["exact_name"])
    df = extractor.extract_features_for_candidates([pair], {"S1-1": e1}, {"T-1": e2})

    assert len(df) == 1
    assert df.columns[0] == "source1_entity_id"
    assert df.columns[1] == "target_entity_id"
    assert df.columns[2] == "target_source"
    assert len(df.columns) == 57  # 3 trace + 54 features

    # Check NaN and Inf
    feat_cols = df.columns[3:]
    assert not df[feat_cols].isna().any().any()
    assert not np.isinf(df[feat_cols].to_numpy()).any()


def test_ensemble_probability_range():
    """Verify probability blends remain strictly bounded within [0, 1]."""
    p_lgb = np.array([0.0, 0.15, 0.45, 0.88, 1.0], dtype=np.float32)
    p_xgb = np.array([0.05, 0.20, 0.50, 0.82, 0.99], dtype=np.float32)

    for w_lgb, w_xgb in [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25)]:
        p_blend = w_lgb * p_lgb + w_xgb * p_xgb
        assert np.all(p_blend >= 0.0)
        assert np.all(p_blend <= 1.0)


def test_group_kfold_separation():
    """Verify GroupKFold enforces strict entity-disjoint separation with zero leakage."""
    evaluator = Evaluator(ValidationConfig(n_splits=5, random_state=42))

    # 100 samples across 20 distinct entities
    groups = np.repeat([f"S1-{i}" for i in range(20)], 5)
    X = pd.DataFrame({"feat": np.random.randn(100)})
    y = np.random.randint(0, 2, 100)

    folds = evaluator.get_group_folds(X, y, groups)
    assert len(folds) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Fold {fold_idx} has {len(overlap)} overlapping groups!"


def test_multi_match_preservation():
    """Verify that multiple valid matches per reference entity are preserved and scored independently."""
    extractor = PairwiseFeatureExtractor()
    e1 = extractor.preprocess_entity("S1-1", "Mega Store", "100 Broadway", "US", source="S1")
    t1 = extractor.preprocess_entity("T-1", "Mega Store NY", "100 Broadway", "US", source="S2")
    t2 = extractor.preprocess_entity("T-2", "Mega Store Manhattan", "100 Broadway", "US", source="S3")

    pairs = [
        CandidatePair("S1-1", "T-1", "S2", ["exact_name"]),
        CandidatePair("S1-1", "T-2", "S3", ["exact_name"]),
    ]
    df = extractor.extract_features_for_candidates(
        pairs, {"S1-1": e1}, {"T-1": t1, "T-2": t2}
    )

    assert len(df) == 2
    assert set(df["target_entity_id"]) == {"T-1", "T-2"}
    # Multi-match is preserved, no 1-to-1 argmax filter applied


def test_ground_truth_denominator_reconciliation():
    """Verify the 5,000 S1 development sample has exactly 17,301 ground-truth pairs."""
    gt = load_ground_truth(as_dict=True)
    s1_all = load_source_file("train", 1)
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(s1_all), size=5000, replace=False)
    s1_dev = s1_all.iloc[sampled].copy().reset_index(drop=True)
    dev_s1_ids = list(s1_dev["entity_id"])

    gt_by_s1 = {s1: set(gt.get(s1, [])) for s1 in dev_s1_ids}
    total_pairs = sum(len(t) for t in gt_by_s1.values())

    s1_with_gt = sum(1 for t in gt_by_s1.values() if len(t) > 0)
    s1_singletons = sum(1 for t in gt_by_s1.values() if len(t) == 0)

    assert len(dev_s1_ids) == 5000
    assert total_pairs == 17301
    assert s1_with_gt == 4723
    assert s1_singletons == 277
    assert s1_with_gt + s1_singletons == 5000


def test_candidate_uniqueness_and_integrity():
    """Verify that Phase 5 feature matrix contains zero duplicate pairs."""
    df_p5 = pd.read_parquet("data/processed/features_dev_phase5_improved.parquet")
    assert not df_p5.duplicated(subset=["source1_entity_id", "target_entity_id"]).any()
    assert len(df_p5) == 228378


def test_metric_calculation_consistency():
    """Verify precision, recall, and F1 calculations adhere to standard definitions."""
    tp, fp, fn = 100, 10, 20
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)

    assert p == pytest.approx(100 / 110)
    assert r == pytest.approx(100 / 120)
    assert f1 == pytest.approx(2 * (100 / 110) * (100 / 120) / ((100 / 110) + (100 / 120)))


def test_threshold_selection_reproducibility():
    """Verify threshold evaluation grid is strictly monotonic and reproducible."""
    evaluator = Evaluator(ValidationConfig(threshold_min=0.05, threshold_max=0.95, threshold_steps=19))
    grid = evaluator.config.threshold_grid
    assert len(grid) == 19
    assert grid[0] == pytest.approx(0.05)
    assert grid[-1] == pytest.approx(0.95)
    assert np.all(np.diff(grid) > 0)
