import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator, CandidatePair
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path

def get_weight(channels, config):
    w = config.channel_weights
    return sum(w.get(ch, 1) for ch in channels)

def main():
    print("Loading data for Cap Audit...")
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

    # Set high cap to get ALL raw candidates
    config = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        use_addr_alpha=True,
        max_candidates_per_entity=10000, 
        max_key_frequency=5000
    )

    generator = CandidateGenerator(config=config)
    print("Building target index (exact Phase 7 replication)...")
    
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"country": r.get("country", "")}
            indexed_s2 += 1

    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"country": r.get("country", "")}
            indexed_s3 += 1
            
    generator.prune_high_frequency_keys()
    
    print("Generating candidates...")
    t0 = time.time()
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    print(f"Generated {len(candidates)} raw candidates in {time.time() - t0:.2f}s")
    
    # Analyze Candidates
    raw_dict = defaultdict(list)
    for c in candidates:
        raw_dict[c.source1_entity_id].append(c)
        
    audit_results = {
        'total_gt_pairs': len(all_gt_pairs),
        'gt_found_raw': 0,
        'gt_retained_capped': 0,
        'entities_affected': 0,
        'entities_all_gt_lost': 0,
        'country_stats': defaultdict(lambda: {'raw': 0, 'capped': 0, 'gt': 0, 'gt_raw': 0, 'gt_capped': 0}),
        'volume_dist': {'0':0, '1-10':0, '11-25':0, '26-50':0, '51-75':0, '76-99':0, '100':0, '101-150':0, '151-250':0, '251-500':0, '500+':0},
        'ties_at_cutoff': 0,
        'gt_lost_at_tie': 0,
        'gt_same_score_as_cutoff': 0
    }
    
    for _, row in s1_dev.iterrows():
        s1_id = row['entity_id']
        ctry = str(row['country']).strip()
        
        gt_set = set(dev_gt.get(s1_id, []))
        audit_results['country_stats'][ctry]['gt'] += len(gt_set)
        
        cands = raw_dict.get(s1_id, [])
        raw_cands = cands
        capped_cands = cands[:100]
        
        raw_tids = set(c.matched_entity_id for c in raw_cands)
        capped_tids = set(c.matched_entity_id for c in capped_cands)
        
        raw_gt = gt_set & raw_tids
        capped_gt = gt_set & capped_tids
        
        audit_results['gt_found_raw'] += len(raw_gt)
        audit_results['gt_retained_capped'] += len(capped_gt)
        
        audit_results['country_stats'][ctry]['gt_raw'] += len(raw_gt)
        audit_results['country_stats'][ctry]['gt_capped'] += len(capped_gt)
        
        if len(raw_gt) > len(capped_gt):
            audit_results['entities_affected'] += 1
            if len(capped_gt) == 0 and len(raw_gt) > 0:
                audit_results['entities_all_gt_lost'] += 1
                
        # Bucket distribution
        L = len(raw_cands)
        if L == 0: audit_results['volume_dist']['0'] += 1
        elif L <= 10: audit_results['volume_dist']['1-10'] += 1
        elif L <= 25: audit_results['volume_dist']['11-25'] += 1
        elif L <= 50: audit_results['volume_dist']['26-50'] += 1
        elif L <= 75: audit_results['volume_dist']['51-75'] += 1
        elif L <= 99: audit_results['volume_dist']['76-99'] += 1
        elif L == 100: audit_results['volume_dist']['100'] += 1
        elif L <= 150: audit_results['volume_dist']['101-150'] += 1
        elif L <= 250: audit_results['volume_dist']['151-250'] += 1
        elif L <= 500: audit_results['volume_dist']['251-500'] += 1
        else: audit_results['volume_dist']['500+'] += 1
        
        # Tie break analysis
        if len(raw_cands) > 100:
            cutoff_cand = raw_cands[99]
            cutoff_score = (get_weight(cutoff_cand.channels, config), len(cutoff_cand.channels))
            
            tied_cands = [c for c in raw_cands if (get_weight(c.channels, config), len(c.channels)) == cutoff_score]
            if len(tied_cands) > 1:
                audit_results['ties_at_cutoff'] += 1
                
            discarded_gt = gt_set & (raw_tids - capped_tids)
            for t in discarded_gt:
                t_cand = next(c for c in raw_cands if c.matched_entity_id == t)
                t_score = (get_weight(t_cand.channels, config), len(t_cand.channels))
                if t_score == cutoff_score:
                    audit_results['gt_same_score_as_cutoff'] += 1
                    audit_results['gt_lost_at_tie'] += 1
                    
    import json
    out_path = ROOT / "output/cap_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print("Cap audit complete.")

if __name__ == "__main__":
    main()
