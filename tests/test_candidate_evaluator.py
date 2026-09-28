"""Unit tests for candidate recall evaluation."""

import pytest
from business_entity_resolution.candidate_generator import CandidatePair
from business_entity_resolution.candidate_evaluator import (
    CandidateEvaluationReport,
    evaluate_candidates,
)


def test_evaluate_candidates_recall_and_reduction():
    # Ground truth:
    # S1-1 -> [S2-10, S3-20]
    # S1-2 -> [S2-30]
    # S1-3 -> [] (singleton)
    # Total GT pairs = 3
    ground_truth = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": ["S2-30"],
        "S1-3": [],
    }

    s1_metadata = {
        "S1-1": {"country": "US"},
        "S1-2": {"country": "India"},
        "S1-3": {"country": "US"},
    }

    # Generated candidates:
    # S1-1 found S2-10 (hit) and S2-99 (false positive candidate), missed S3-20 (FN)
    # S1-2 found S2-30 (hit)
    # S1-3 found none
    candidates = [
        CandidatePair("S1-1", "S2-10", "S2", ["exact_name"]),
        CandidatePair("S1-1", "S2-99", "S2", ["name_token_shingles"]),
        CandidatePair("S1-2", "S2-30", "S2", ["country_core_name"]),
    ]

    report = evaluate_candidates(
        candidates=candidates,
        ground_truth=ground_truth,
        s1_metadata=s1_metadata,
        total_target_records=100,
    )

    assert report.total_gt_pairs == 3
    assert report.total_candidate_pairs == 3
    assert report.found_gt_pairs == 2
    assert report.missing_gt_pairs == 1
    # Recall = 2 / 3 = 0.6667
    assert abs(report.pair_recall - 0.6667) < 0.001

    # Country breakdown:
    # US had 2 pairs (S2-10, S3-20), found 1 -> 50%
    # India had 1 pair (S2-30), found 1 -> 100%
    assert abs(report.recall_by_country["US"] - 0.5) < 0.001
    assert abs(report.recall_by_country["India"] - 1.0) < 0.001

    # Channels contribution
    assert report.channel_contribution["exact_name"] == 1
    assert report.channel_contribution["country_core_name"] == 1

    # Reduction ratio: total_possible = 3 * 100 = 300; 3 / 300 = 0.01; reduction = 0.99
    assert abs(report.reduction_ratio - 0.99) < 0.001

    # Markdown format check
    md = report.to_markdown()
    assert "Candidate Pair Recall Ceiling" in md
