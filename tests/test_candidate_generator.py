"""Unit tests for candidate generation and source restrictions."""

import pytest
import pandas as pd
from business_entity_resolution.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
)
from business_entity_resolution.preprocessing_config import BlockingConfig


def test_source_restrictions_enforced():
    generator = CandidateGenerator()

    # Indexing an S1 ID as a target record MUST fail (S1 is reference only)
    with pytest.raises(ValueError, match="Source 1 record"):
        generator.index_target_record("S1-100", "Bad Target", "123 St", "US")

    # Indexing S2 and S3 must succeed
    generator.index_target_record("S2-200", "Good S2 Target", "456 Ave", "US")
    generator.index_target_record("S3-300", "Good S3 Target", "789 Rd", "India")
    assert generator.indexed_target_count == 2


def test_candidate_retrieval_and_provenance(tmp_path):
    generator = CandidateGenerator()

    # Index target records from S2 and S3
    generator.index_target_record(
        "S2-001", "Global Logistics Inc", "100 Highway, Dallas, TX 75201", "US"
    )
    generator.index_target_record(
        "S3-002", "Global Logistics Corporation", "200 Freeway, Dallas, TX 75201", "US"
    )
    generator.index_target_record(
        "S2-099", "Unrelated Bakery", "10 Main St, Chicago, IL", "US"
    )

    # Query with a Source 1 record
    candidates = generator.generate_candidates_for_record(
        s1_id="S1-500",
        name="Global Logistics LLC",
        address="100 Highway, Dallas, TX 75201",
        country="US",
    )

    matched_target_ids = {c.matched_entity_id for c in candidates}
    assert "S2-001" in matched_target_ids
    assert "S3-002" in matched_target_ids
    assert "S2-099" not in matched_target_ids

    # Check channels provenance
    c1 = next(c for c in candidates if c.matched_entity_id == "S2-001")
    assert c1.matched_source == "S2"
    # Found by core name or shingles or postal
    assert len(c1.channels) > 0

    # Test official candidate pairs format export
    out_file = tmp_path / "candidate_pairs.tsv"
    CandidateGenerator.export_to_official_candidate_pairs(
        candidates, ["S1-500", "S1-999"], out_file
    )

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    # S1-500 should have comma-separated targets
    assert "S1-500\t" in lines[1]
    assert "S2-001" in lines[1]
    assert "S3-002" in lines[1]
    # S1-999 should be present with empty candidates (singleton)
    assert lines[2] == "S1-999\t"


def test_candidate_cap_and_prioritization():
    # Test that max_candidates_per_entity limits output and prioritizes high-confidence channels
    cfg = BlockingConfig(max_candidates_per_entity=2)
    generator = CandidateGenerator(config=cfg)

    # Index 4 targets:
    # S2-10: exact match
    # S2-20: address stem match
    # S2-30: shingle match
    # S2-40: shingle match
    generator.index_target_record("S2-10", "Apex Data Systems", "100 Market St", "US")
    generator.index_target_record("S2-20", "Completely Different Name", "100 Market St", "US")
    generator.index_target_record("S2-30", "Apex Global Trading", "999 Other St", "US")
    generator.index_target_record("S2-40", "Apex Industrial Solutions", "888 Another Rd", "US")

    candidates = generator.generate_candidates_for_record(
        s1_id="S1-1",
        name="Apex Data Systems",
        address="100 Market St",
        country="US",
    )

    # Must be capped at exactly 2
    assert len(candidates) == 2
    cand_ids = [c.matched_entity_id for c in candidates]
    # S2-10 has exact name match (highest weight)
    assert "S2-10" in cand_ids
    # S2-20 has address match (high weight)
    assert "S2-20" in cand_ids


def test_duplicate_candidate_handling():
    generator = CandidateGenerator()

    # S2-1 shares exact name, core name, and address stem with S1
    generator.index_target_record("S2-1", "Acme Logistics Inc", "123 Main St", "US")

    candidates = generator.generate_candidates_for_record(
        s1_id="S1-1",
        name="Acme Logistics Inc",
        address="123 Main St",
        country="US",
    )

    # Must have exactly 1 record for S2-1, not multiple duplicates
    assert len(candidates) == 1
    assert candidates[0].matched_entity_id == "S2-1"
    # Channel provenance must contain all triggered channels
    assert len(candidates[0].channels) > 1
    assert "exact_name" in candidates[0].channels

