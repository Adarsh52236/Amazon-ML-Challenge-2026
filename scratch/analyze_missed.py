import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path
from business_entity_resolution.config import configure_console_encoding

configure_console_encoding()

def analyze_missed_pairs():
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

    config = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        max_candidates_per_entity=None,
        max_key_frequency=5000
    )

    generator = CandidateGenerator(config=config)
    
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            if eid in target_ids_needed:
                raw_targets[eid] = r
            indexed_s2 += 1

    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            if eid in target_ids_needed:
                raw_targets[eid] = r
            indexed_s3 += 1
            
    generator.prune_high_frequency_keys()
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    candidate_set = set((c.source1_entity_id, c.matched_entity_id) for c in candidates)
    
    missed = all_gt_pairs - candidate_set
    print(f"Total ground-truth pairs missed: {len(missed)}")
    
    s1_dict = s1_dev.set_index("entity_id").to_dict("index")
    
    print("\n--- SAMPLE MISSED PAIRS ---")
    for i, (s1_id, tid) in enumerate(list(missed)[:15]):
        s1_row = s1_dict[s1_id]
        t_row = raw_targets[tid]
        print(f"Pair {i+1}:")
        print(f"  S1: {s1_row['business_name']} | {s1_row.get('business_address', '')} | {s1_row.get('country', '')}")
        print(f"  T : {t_row.get('business_name', '')} | {t_row.get('business_address', '')} | {t_row.get('country', '')}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_missed_pairs()
