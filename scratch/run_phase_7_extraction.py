import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

def main():
    print("Loading data...")
    seed = 42
    sample_size = 5000
    target_scan_limit = 100_000

    gt_mapping = load_ground_truth(as_dict=True)
    s1_all = load_source_file("train", 1)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(s1_all), size=sample_size, replace=False)
    s1_dev = s1_all.iloc[sampled].copy().reset_index(drop=True)
    dev_s1_ids = set(s1_dev["entity_id"])

    dev_gt = {s1_id: gt_mapping.get(s1_id, []) for s1_id in dev_s1_ids if s1_id in gt_mapping}
    all_gt_pairs = set()
    for s1_id, targets in dev_gt.items():
        for t in targets:
            all_gt_pairs.add((s1_id, t))

    target_ids_needed = {t for _, t in all_gt_pairs}
    print(f"Total ground-truth positive pairs for sample: {len(all_gt_pairs)}")

    config = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        use_addr_alpha=True,
        max_candidates_per_entity=250,
        max_key_frequency=5000
    )

    generator = CandidateGenerator(config=config)
    print("Building target index...")
    
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
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
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
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
    
    print("Generating candidates...")
    t0 = time.time()
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    print(f"Generated {len(candidates)} candidates in {time.time() - t0:.2f}s")
    
    print("Extracting features...")
    t0 = time.time()
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)

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
        
    feature_df = extractor.extract_features_for_candidates(
        candidates, s1_preprocessed, target_preprocessed
    )
    print(f"Extracted features for {len(feature_df)} pairs in {time.time() - t0:.2f}s")
    
    out_path = Path("data/processed/features_dev_phase5_improved.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
