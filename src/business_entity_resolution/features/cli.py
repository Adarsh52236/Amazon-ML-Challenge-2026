"""CLI entry point for running pairwise feature extraction."""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.config import (
    configure_console_encoding,
    get_source_path,
)
from business_entity_resolution.data_loader import iter_source_records, load_source_file
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
from business_entity_resolution.preprocessing_config import BlockingConfig

configure_console_encoding()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pairwise features for candidate pairs.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 reference entities.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation split.")
    parser.add_argument("--cap", type=int, default=100, help="Candidate cap per entity (default: 100).")
    parser.add_argument(
        "--output",
        type=str,
        default="output/features_sample.parquet",
        help="Output path for feature matrix (.parquet or .tsv).",
    )
    parser.add_argument("--target-scan-limit", type=int, default=100_000, help="Scan limit per target source.")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 3: PAIRWISE FEATURE EXTRACTION")
    print(f"Sample: {args.sample_size:,} S1 entities | Seed: {args.seed} | Cap: {args.cap}")
    print(f"Output: {args.output}")
    print("=" * 80)

    tracemalloc.start()
    t_start = time.time()

    # 1. Load S1 reference sample
    s1_all = load_source_file("train", 1)
    rng = np.random.default_rng(args.seed)
    sampled = rng.choice(len(s1_all), size=args.sample_size, replace=False)
    s1_dev = s1_all.iloc[sampled].copy().reset_index(drop=True)
    dev_s1_ids = set(s1_dev["entity_id"])

    # 2. Build Candidate Generator & Index Targets
    b_cfg = BlockingConfig(max_candidates_per_entity=args.cap)
    generator = CandidateGenerator(config=b_cfg)

    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if indexed_s2 < args.target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {
                "entity_id": eid,
                "name": r.get("business_name", ""),
                "address": r.get("business_address", ""),
                "country": r.get("country", ""),
                "source": "S2",
            }
            indexed_s2 += 1

    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if indexed_s3 < args.target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {
                "entity_id": eid,
                "name": r.get("business_name", ""),
                "address": r.get("business_address", ""),
                "country": r.get("country", ""),
                "source": "S3",
            }
            indexed_s3 += 1

    generator.prune_high_frequency_keys()
    print(f"Indexed {generator.indexed_target_count:,} target records.")

    # 3. Retrieve Candidate Pairs
    t_cand = time.time()
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    cand_time = time.time() - t_cand
    print(f"Generated {len(candidates):,} candidate pairs in {cand_time:.2f}s.")

    # 4. Initialize Feature Extractor & Preprocess Entities
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)

    print("Preprocessing unique entities...")
    t_prep = time.time()
    s1_preprocessed = {}
    for _, row in s1_dev.iterrows():
        eid = str(row["entity_id"])
        s1_preprocessed[eid] = extractor.preprocess_entity(
            entity_id=eid,
            name=row.get("business_name", ""),
            address=row.get("business_address", ""),
            country=row.get("country", ""),
            source="S1",
        )

    # Only preprocess target entities that appear in candidates
    candidate_target_ids = {c.matched_entity_id for c in candidates}
    target_preprocessed = {}
    for tid in candidate_target_ids:
        r = raw_targets[tid]
        target_preprocessed[tid] = extractor.preprocess_entity(
            entity_id=tid,
            name=r["name"],
            address=r["address"],
            country=r["country"],
            source=r["source"],
        )
    prep_time = time.time() - t_prep
    print(f"Preprocessed {len(s1_preprocessed):,} S1 and {len(target_preprocessed):,} target entities in {prep_time:.2f}s.")

    # 5. Extract Feature Matrix
    print("Extracting pairwise feature matrix...")
    t_feat = time.time()
    feature_df = extractor.extract_features_for_candidates(
        candidates, s1_preprocessed, target_preprocessed
    )
    feat_time = time.time() - t_feat
    total_time = time.time() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Extracted {feature_df.shape[1] - 3} features across {len(feature_df):,} pairs in {feat_time:.2f}s.")
    print(f"Total time: {total_time:.2f}s | Peak memory: {peak_mem / (1024 * 1024):.2f} MB")

    # 6. Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        feature_df.to_parquet(out_path, index=False)
    else:
        feature_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved feature matrix to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
