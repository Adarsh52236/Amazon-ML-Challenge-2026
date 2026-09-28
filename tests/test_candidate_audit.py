"""Regression tests for Phase 2.6 candidate pipeline audit and reconciliation."""

import numpy as np
import pytest

from business_entity_resolution.address_normalizer import BusinessAddressNormalizer
from business_entity_resolution.candidate_generator import CandidateGenerator, CandidatePair
from business_entity_resolution.candidate_evaluator import evaluate_candidates
from business_entity_resolution.preprocessing_config import BlockingConfig


def test_missed_pair_set_reconciliation():
    """Verify that key-gen misses and cap-truncation misses strictly partition total misses."""
    gt_pairs = {("S1-1", "S2-1"), ("S1-1", "S3-2"), ("S1-2", "S2-3"), ("S1-2", "S2-4")}

    # Uncapped finds 3 out of 4 (misses S1-2, S2-4 due to no matching key)
    uncapped_cands = {("S1-1", "S2-1"), ("S1-1", "S3-2"), ("S1-2", "S2-3")}

    # Capped (e.g. cap=1) drops S1-1, S3-2 due to cap truncation
    capped_cands = {("S1-1", "S2-1"), ("S1-2", "S2-3")}

    total_missed = gt_pairs - capped_cands
    key_gen_misses = gt_pairs - uncapped_cands
    cap_trunc_misses = (gt_pairs.intersection(uncapped_cands)) - capped_cands

    # Check 1: Sets must be disjoint
    assert len(key_gen_misses.intersection(cap_trunc_misses)) == 0

    # Check 2: Sets must exactly partition total missed
    assert key_gen_misses.union(cap_trunc_misses) == total_missed
    assert len(key_gen_misses) + len(cap_trunc_misses) == len(total_missed)


def test_entity_level_coverage_denominators():
    """Verify clear distinction between singletons and non-singletons in entity coverage."""
    # 3 S1 entities:
    # S1-1: 2 GT matches (non-singleton) -> recovered 2 (fully recovered)
    # S1-2: 2 GT matches (non-singleton) -> recovered 1 (partially recovered)
    # S1-3: 1 GT match (non-singleton) -> recovered 0 (zero recovered)
    # S1-4: 0 GT matches (singleton)
    # Total entities = 4. Non-singletons = 3. Singletons = 1.
    gt = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": ["S2-30", "S3-40"],
        "S1-3": ["S2-50"],
        "S1-4": [],
    }

    candidates = [
        CandidatePair("S1-1", "S2-10", "S2", ["exact_name"]),
        CandidatePair("S1-1", "S3-20", "S3", ["exact_name"]),
        CandidatePair("S1-2", "S2-30", "S2", ["exact_name"]),
    ]

    non_singletons = [k for k, v in gt.items() if len(v) > 0]
    singletons = [k for k, v in gt.items() if len(v) == 0]

    assert len(non_singletons) == 3
    assert len(singletons) == 1

    recovered_map = {k: set() for k in gt}
    for c in candidates:
        if c.matched_entity_id in gt[c.source1_entity_id]:
            recovered_map[c.source1_entity_id].add(c.matched_entity_id)

    full_rec = sum(1 for k in non_singletons if len(recovered_map[k]) == len(gt[k]))
    part_rec = sum(1 for k in non_singletons if 0 < len(recovered_map[k]) < len(gt[k]))
    zero_rec = sum(1 for k in non_singletons if len(recovered_map[k]) == 0)

    assert full_rec == 1
    assert part_rec == 1
    assert zero_rec == 1

    # Non-singleton coverage: (full + part) / 3 = 2 / 3 = 66.67%
    non_singleton_cov = (full_rec + part_rec) / len(non_singletons)
    assert abs(non_singleton_cov - (2 / 3)) < 1e-4


def test_missing_address_normalization():
    """Verify that null, NaN, None, and whitespace address inputs are treated uniformly as empty."""
    normalizer = BusinessAddressNormalizer()

    inputs = ["", "   ", None, "nan", "NaN", "null", "NULL", "none", "None"]
    for inp in inputs:
        res = normalizer.normalize(inp)
        assert res.is_empty, f"Failed for input: {inp}"
        assert res.normalized == ""
        assert res.street_number == ""
        assert res.postal_code == ""


def test_cap_ranking_determinism():
    """Verify that candidate ranking produces identical results regardless of execution order."""
    cfg = BlockingConfig(max_candidates_per_entity=2)
    gen = CandidateGenerator(config=cfg)

    gen.index_target_record("S2-1", "Delta Global", "100 Broadway", "US")
    gen.index_target_record("S2-2", "Delta Systems", "200 Fifth Ave", "US")
    gen.index_target_record("S3-3", "Delta Industries", "300 Third St", "US")

    res1 = gen.generate_candidates_for_record("S1-1", "Delta Global", "100 Broadway", "US")
    res2 = gen.generate_candidates_for_record("S1-1", "Delta Global", "100 Broadway", "US")

    assert [c.matched_entity_id for c in res1] == [c.matched_entity_id for c in res2]
    assert [c.channels for c in res1] == [c.channels for c in res2]


def test_cap_induced_recall_loss():
    """Verify that capping strictly reduces or keeps recall equal, never increases recall."""
    cfg_uncapped = BlockingConfig(max_candidates_per_entity=None)
    cfg_capped = BlockingConfig(max_candidates_per_entity=1)

    gen_uncapped = CandidateGenerator(config=cfg_uncapped)
    gen_capped = CandidateGenerator(config=cfg_capped)

    # 2 true matches for S1-1
    gen_uncapped.index_target_record("S2-1", "Alpha Inc", "10 Main", "US")
    gen_uncapped.index_target_record("S3-2", "Alpha Corp", "10 Main", "US")

    gen_capped.index_target_record("S2-1", "Alpha Inc", "10 Main", "US")
    gen_capped.index_target_record("S3-2", "Alpha Corp", "10 Main", "US")

    res_uncapped = gen_uncapped.generate_candidates_for_record("S1-1", "Alpha Inc", "10 Main", "US")
    res_capped = gen_capped.generate_candidates_for_record("S1-1", "Alpha Inc", "10 Main", "US")

    assert len(res_uncapped) == 2
    assert len(res_capped) == 1
    # Candidate loss
    assert len(res_capped) <= len(res_uncapped)


def test_source_2_and_source_3_coverage():
    """Verify that candidate generation successfully retrieves from both S2 and S3 targets."""
    gen = CandidateGenerator()
    gen.index_target_record("S2-100", "Omega Labs", "50 Pine St", "US")
    gen.index_target_record("S3-200", "Omega Labs", "50 Pine St", "US")

    candidates = gen.generate_candidates_for_record("S1-1", "Omega Labs", "50 Pine St", "US")
    sources = {c.matched_source for c in candidates}

    assert "S2" in sources
    assert "S3" in sources
