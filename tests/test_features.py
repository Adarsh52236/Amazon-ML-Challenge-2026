"""Unit and regression tests for Phase 3 pairwise feature extraction."""

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.address_normalizer import BusinessAddressNormalizer
from business_entity_resolution.candidate_generator import CandidatePair
from business_entity_resolution.features.address_features import extract_address_features
from business_entity_resolution.features.country_features import extract_country_features
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.name_features import extract_name_features
from business_entity_resolution.features.pairwise_extractor import (
    PairwiseFeatureExtractor,
    PreprocessedEntity,
)
from business_entity_resolution.features.provenance_features import extract_provenance_features
from business_entity_resolution.features.similarity_metrics import (
    char_ngram_jaccard,
    difflib_similarity,
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_containment,
    token_jaccard,
)
from business_entity_resolution.features.source_features import extract_source_features
from business_entity_resolution.text_normalizer import BusinessNameNormalizer


def test_similarity_metrics_correctness():
    # Exact equality
    assert levenshtein_similarity("acme", "acme") == 1.0
    assert jaro_winkler_similarity("acme", "acme") == 1.0
    assert char_ngram_jaccard("acme", "acme") == 1.0
    assert token_jaccard(["acme", "corp"], ["acme", "corp"]) == 1.0
    assert token_containment(["acme"], ["acme", "corp"]) == 1.0

    # Completely different
    assert levenshtein_similarity("acme", "xyz") == 0.0
    assert char_ngram_jaccard("abc", "xyz") == 0.0
    assert token_jaccard(["acme"], ["xyz"]) == 0.0

    # Partial / typo
    assert levenshtein_similarity("microsoft", "microsft") > 0.85
    assert jaro_winkler_similarity("dwayne", "duane") > 0.80


def test_name_features_exact_and_diacritics():
    norm = BusinessNameNormalizer()
    cfg = FeatureConfig()

    n1 = norm.normalize("Kripa Polytechnic")
    n2 = norm.normalize("Shri Kripa Pólytechnic")

    feats = extract_name_features(n1, n2, cfg)

    # Thanks to diacritic stripping, polytechnic matches polytechnic
    assert feats["name_token_overlap_count"] >= 2.0  # kripa, polytechnic
    assert feats["name_last_token_match"] == 1.0
    assert feats["name_jaro_winkler_sim"] > 0.75
    assert feats["name_levenshtein_sim"] > 0.70
    assert feats["name_token_jaccard"] > 0.50


def test_name_features_empty_handling():
    norm = BusinessNameNormalizer()
    cfg = FeatureConfig()

    n_empty = norm.normalize("")
    n_valid = norm.normalize("Acme Corp")

    feats = extract_name_features(n_empty, n_valid, cfg)
    # Check that all features are valid non-NaN numbers
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is infinite"
    assert feats["name_exact_match"] == 0.0
    assert feats["name_levenshtein_sim"] == 0.0


def test_address_features_structured_components():
    addr_norm = BusinessAddressNormalizer()
    cfg = FeatureConfig()

    a1 = addr_norm.normalize("100 Market St, San Francisco, CA 94105", country="US")
    a2 = addr_norm.normalize("100 Market Street, Suite 200, San Francisco, CA 94105-1234", country="US")

    feats = extract_address_features(a1, a2, country="US", config=cfg)

    assert feats["address_street_number_match"] == 1.0
    assert feats["address_street_number_both_present"] == 1.0
    assert feats["address_street_number_mismatch"] == 0.0
    assert feats["address_postal_exact_match"] == 1.0
    assert feats["address_postal_prefix_match"] == 1.0
    assert feats["address_street_stem_match"] == 1.0
    assert feats["address_both_present"] == 1.0
    assert feats["address_token_jaccard"] > 0.50


def test_address_features_missing_handling():
    addr_norm = BusinessAddressNormalizer()
    cfg = FeatureConfig()

    a_valid = addr_norm.normalize("123 Main St", country="US")
    a_empty = addr_norm.normalize(None, country="US")

    feats = extract_address_features(a_valid, a_empty, country="US", config=cfg)

    assert feats["address_missing_source"] == 0.0
    assert feats["address_missing_target"] == 1.0
    assert feats["address_both_missing"] == 0.0
    assert feats["address_both_present"] == 0.0
    assert feats["address_street_number_match"] == 0.0
    assert feats["address_token_overlap_count"] == 0.0
    for k, v in feats.items():
        assert not np.isnan(v)
        assert not np.isinf(v)


def test_country_features_agreement_and_conflict():
    cfg = FeatureConfig()

    # Match
    f_match = extract_country_features("US", "US", cfg)
    assert f_match["country_exact_match"] == 1.0
    assert f_match["country_conflict"] == 0.0

    # Conflict
    f_conf = extract_country_features("US", "India", cfg)
    assert f_conf["country_exact_match"] == 0.0
    assert f_conf["country_conflict"] == 1.0

    # Missing target
    f_miss = extract_country_features("US", "", cfg)
    assert f_miss["country_missing_target"] == 1.0
    assert f_miss["country_exact_match"] == 0.0
    assert f_conf["country_conflict"] == 1.0


def test_provenance_and_source_features():
    cfg = FeatureConfig()

    prov = extract_provenance_features(["exact_name", "addr_num_stem"], cfg)
    assert prov["prov_channel_count"] == 2.0
    assert prov["prov_max_channel_weight"] == 10.0
    assert prov["prov_has_exact_name"] == 1.0
    assert prov["prov_has_addr_num_stem"] == 1.0
    assert prov["prov_has_postal_prefix"] == 0.0

    src_s2 = extract_source_features("S2", cfg)
    assert src_s2["target_source_is_s2"] == 1.0
    assert src_s2["target_source_is_s3"] == 0.0

    src_s3 = extract_source_features("S3", cfg)
    assert src_s3["target_source_is_s2"] == 0.0
    assert src_s3["target_source_is_s3"] == 1.0


def test_pairwise_extractor_end_to_end_and_determinism():
    extractor = PairwiseFeatureExtractor()
    cfg = FeatureConfig()

    e1 = extractor.preprocess_entity(
        "S1-1", "Apex Logistics LLC", "100 Broadway, New York, NY 10001", "US", source="S1"
    )
    e2 = extractor.preprocess_entity(
        "S2-2", "Apex Logistics Inc", "100 Broadway, New York, NY 10001", "US", source="S2"
    )

    pairs = [
        CandidatePair("S1-1", "S2-2", "S2", ["exact_name", "addr_num_stem"])
    ]

    s1_dict = {"S1-1": e1}
    target_dict = {"S2-2": e2}

    df1 = extractor.extract_features_for_candidates(pairs, s1_dict, target_dict)
    df2 = extractor.extract_features_for_candidates(pairs, s1_dict, target_dict)

    # 1. Check schema consistency
    expected_cols = ["source1_entity_id", "target_entity_id", "target_source"] + cfg.get_feature_names()
    assert list(df1.columns) == expected_cols

    # 2. Check no NaNs
    assert df1.isna().sum().sum() == 0

    # 3. Check determinism (identical DataFrame values)
    pd.testing.assert_frame_equal(df1, df2)

    # 4. Check feature values
    row = df1.iloc[0]
    assert row["name_core_exact_match"] == 1.0
    assert row["address_postal_exact_match"] == 1.0
    assert row["country_exact_match"] == 1.0
    assert row["target_source_is_s2"] == 1.0
    assert row["prov_has_exact_name"] == 1.0


def test_no_ground_truth_leakage():
    """Verify that feature extraction works without any ground-truth input."""
    extractor = PairwiseFeatureExtractor()

    e1 = extractor.preprocess_entity("S1-10", "Test Corp", "10 Elm St", "US")
    e2 = extractor.preprocess_entity("S3-20", "Test Inc", "10 Elm St", "US")

    pair = CandidatePair("S1-10", "S3-20", "S3", ["name_token_shingles"])

    # Does not require ground-truth
    df = extractor.extract_features_for_candidates([pair], {"S1-10": e1}, {"S3-20": e2})

    assert "ground_truth" not in df.columns
    assert "label" not in df.columns
    assert "matched" not in df.columns
    assert len(df) == 1
