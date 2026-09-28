import os
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path

def evaluate_candidates():
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

    # Test config with NO CAPPING
    config = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        use_addr_alpha=True,
        max_candidates_per_entity=250, # Capped at 250
        max_key_frequency=5000 # Bump up to 5000
    )

    generator = CandidateGenerator(config=config)
    print("Building target index...")
    
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            indexed_s2 += 1

    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            indexed_s3 += 1
            
    pruned = generator.prune_high_frequency_keys()
    print(f"Pruned keys: {pruned}")
    
    print("Generating candidates...")
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    
    print(f"Total candidate pairs generated: {len(candidates)}")
    print(f"Average candidates per S1 entity: {len(candidates) / len(s1_dev):.2f}")
    
    candidate_set = set((c.source1_entity_id, c.matched_entity_id) for c in candidates)
    
    found = candidate_set.intersection(all_gt_pairs)
    missed = all_gt_pairs - candidate_set
    
    recall = len(found) / len(all_gt_pairs) if all_gt_pairs else 0
    print(f"Ground-truth pairs found: {len(found)}")
    print(f"Ground-truth pairs missed: {len(missed)}")
    print(f"Candidate Recall: {recall:.4f} ({recall*100:.2f}%)")

if __name__ == "__main__":
    evaluate_candidates()
