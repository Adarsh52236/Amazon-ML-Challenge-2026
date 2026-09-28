import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import iter_source_records, get_source_path, load_source_file, load_ground_truth

def get_weight(channels, config):
    weights = getattr(config, "channel_weights", {}) or {}
    return sum(weights.get(ch, 1) for ch in channels)

def main():
    print("Loading dev sample and GT...")
    feat_path = ROOT / "data/processed/features_dev_phase5_improved.parquet"
    feat_df = pd.read_parquet(feat_path, columns=["source1_entity_id"])
    dev_s1_ids = set(feat_df["source1_entity_id"])
    
    # We need the full dataframe to get country info
    s1_all = load_source_file("train", 1)
    s1_dev = s1_all[s1_all["entity_id"].isin(dev_s1_ids)].copy().reset_index(drop=True)
    
    from business_entity_resolution.preprocessing_config import BlockingConfig
    config = BlockingConfig()
    
    gt_df = load_ground_truth()
    
    dev_gt = defaultdict(set)
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        if s1 in dev_s1_ids:
            dev_gt[s1] = set(row['matched_entity_ids'].split(','))
            
    all_gt_pairs = []
    for s1, tgts in dev_gt.items():
        for t in tgts:
            all_gt_pairs.append((s1, t))
            
    print(f"Loaded {len(dev_s1_ids)} S1 entities and {len(all_gt_pairs)} GT pairs.")
    
    target_ids_needed = set()
    for tgts in dev_gt.values():
        target_ids_needed.update(tgts)
        
    print("Initializing blocker...")
        
    config.max_candidates_per_entity = 10000
    config.max_key_frequency = 5000
    generator = CandidateGenerator(config=config)
    
    print("Indexing S2 and S3...")
    target_scan_limit = 100000 # Same as before probably, or all? Let's just index all required
    # Wait, the previous script did: `if eid in target_ids_needed or indexed_s2 < target_scan_limit:`
    # Let's reproduce that exactly
    
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
            
    generator.prune_high_frequency_keys()
    
    print("Generating raw candidates...")
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    
    raw_dict = defaultdict(list)
    for c in candidates:
        raw_dict[c.source1_entity_id].append(c)
        
    # Total candidates:
    total_cands = sum(len(c) for c in raw_dict.values())
    print(f"Total raw candidates: {total_cands}")
    
    country_map = dict(zip(s1_dev['entity_id'], s1_dev['country']))
    
    # 2. DEFINE CAP-LOSS GT PAIRS
    print("Finding cap-lost GT pairs...")
    
    lost_gt_info = []
    
    # We will simulate exactly the cap=100 logic (top 100 per S1, using existing ranking order in `cands`)
    gt_present_uncapped = 0
    for s1_id in dev_s1_ids:
        gt_set = set(dev_gt.get(s1_id, []))
        cands = raw_dict.get(s1_id, [])
        ctry = str(country_map[s1_id]).strip()
        
        raw_tids = [c.matched_entity_id for c in cands]
        
        for t in gt_set:
            if t in raw_tids:
                gt_present_uncapped += 1
                rank = raw_tids.index(t) + 1
                if rank > 100:
                    cand = cands[rank - 1]
                    priority_score = get_weight(cand.channels, config)
                    lost_gt_info.append({
                        's1_id': s1_id,
                        'target_id': t,
                        'country': ctry,
                        'uncapped_cands': len(cands),
                        'raw_rank': rank,
                        'priority_score': priority_score,
                        'channel_count': len(cand.channels),
                        'channels': list(cand.channels),
                        'cand_obj': cand,
                        'cands_list': cands
                    })
                    
    print(f"GT Pairs present uncapped: {gt_present_uncapped}")
    print(f"Lost GT Pairs at Cap 100: {len(lost_gt_info)}")
    
    # 3. RANK DISTRIBUTION
    rank_buckets = {'1-10':0, '11-25':0, '26-50':0, '51-75':0, '76-100':0, '101-150':0, '151-200':0, '201-300':0, '301-500':0, '501-1000':0, '1000+':0}
    for lost in lost_gt_info:
        r = lost['raw_rank']
        if r <= 10: rank_buckets['1-10'] += 1
        elif r <= 25: rank_buckets['11-25'] += 1
        elif r <= 50: rank_buckets['26-50'] += 1
        elif r <= 75: rank_buckets['51-75'] += 1
        elif r <= 100: rank_buckets['76-100'] += 1
        elif r <= 150: rank_buckets['101-150'] += 1
        elif r <= 200: rank_buckets['151-200'] += 1
        elif r <= 300: rank_buckets['201-300'] += 1
        elif r <= 500: rank_buckets['301-500'] += 1
        elif r <= 1000: rank_buckets['501-1000'] += 1
        else: rank_buckets['1000+'] += 1
        
    print("Rank Buckets for LOST GT:", rank_buckets)
    
    # Let's save the data to a pickle or json so we can analyze it without running CandidateGenerator again
    import pickle
    with open(ROOT / "output/phase6_audit_data.pkl", "wb") as f:
        pickle.dump({
            'raw_dict': raw_dict,
            'dev_gt': dev_gt,
            'all_gt_pairs': all_gt_pairs,
            'country_map': country_map,
            'lost_gt_info': [{k:v for k,v in l.items() if k not in ['cand_obj', 'cands_list']} for l in lost_gt_info],
            'config': config.__dict__
        }, f)
        
    print("Data saved. Run analysis script.")

if __name__ == "__main__":
    main()
