import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def get_weight(channels, config):
    weights = config.get("channel_weights", {}) or {}
    return sum(weights.get(ch, 1) for ch in channels)

def main():
    print("Loading data...")
    with open(ROOT / "output/phase6_audit_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    raw_dict = data['raw_dict']
    dev_gt = data['dev_gt']
    all_gt_pairs = data['all_gt_pairs']
    country_map = data['country_map']
    lost_gt_info = data['lost_gt_info']
    config = data['config']
    
    dev_s1_ids = list(dev_gt.keys())
    
    # 1. RECREATE PRE-CAP SET
    total_raw_candidates = sum(len(c) for c in raw_dict.values())
    total_gt = len(all_gt_pairs)
    
    raw_gt_present = 0
    cands_per_s1 = []
    cands_per_country = defaultdict(list)
    
    for s1, cands in raw_dict.items():
        ctry = country_map[s1]
        cands_per_s1.append(len(cands))
        cands_per_country[ctry].append(len(cands))
        
        tids = set(c.matched_entity_id for c in cands)
        gt = set(dev_gt.get(s1, []))
        raw_gt_present += len(tids & gt)
        
    print(f"Total raw candidates: {total_raw_candidates}")
    print(f"Total GT pairs: {total_gt}")
    print(f"Total GT present before cap: {raw_gt_present} ({raw_gt_present/total_gt:.2%})")
    
    # 2 & 3. RANK DISTRIBUTION OF LOST TRUE MATCHES
    # Already computed in generation script, let's recount
    rank_buckets = {'1-10':0, '11-25':0, '26-50':0, '51-75':0, '76-100':0, '101-150':0, '151-200':0, '201-300':0, '301-500':0, '501-1000':0, '1000+':0}
    country_lost = defaultdict(int)
    lost_cands_count = []
    
    for lost in lost_gt_info:
        r = lost['raw_rank']
        ctry = lost['country']
        country_lost[ctry] += 1
        lost_cands_count.append(lost['uncapped_cands'])
        
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
        
    print(f"Total Lost GT at cap 100: {len(lost_gt_info)}")
    
    # 4. CHANNEL ANALYSIS
    channel_recovered = defaultdict(int)
    channel_lost = defaultdict(int)
    channel_lost_priority = defaultdict(list)
    channel_lost_country = defaultdict(lambda: defaultdict(int))
    
    # Need to get channels for all recovered GT pairs to compare
    recovered_gt_count = 0
    strong_channel_support = {'recovered': 0, 'lost': 0}
    weak_channel_only = {'recovered': 0, 'lost': 0}
    multi_channel = {'recovered': 0, 'lost': 0}
    single_channel = {'recovered': 0, 'lost': 0}
    
    def is_strong(ch):
        # Infer strong channels: weights >= some value, e.g. 1.0 or based on config
        w = config['channel_weights'].get(ch, 0.5)
        return w >= 1.0

    for s1, cands in raw_dict.items():
        gt = set(dev_gt.get(s1, []))
        for i, c in enumerate(cands):
            if c.matched_entity_id in gt:
                if i < 100:
                    recovered_gt_count += 1
                    for ch in c.channels:
                        channel_recovered[ch] += 1
                    
                    has_strong = any(is_strong(ch) for ch in c.channels)
                    if has_strong: strong_channel_support['recovered'] += 1
                    else: weak_channel_only['recovered'] += 1
                    
                    if len(c.channels) > 1: multi_channel['recovered'] += 1
                    else: single_channel['recovered'] += 1
                    
    for lost in lost_gt_info:
        for ch in lost['channels']:
            channel_lost[ch] += 1
            channel_lost_priority[ch].append(config['channel_weights'].get(ch, 0.5))
            channel_lost_country[ch][lost['country']] += 1
            
        has_strong = any(is_strong(ch) for ch in lost['channels'])
        if has_strong: strong_channel_support['lost'] += 1
        else: weak_channel_only['lost'] += 1
        
        if len(lost['channels']) > 1: multi_channel['lost'] += 1
        else: single_channel['lost'] += 1
        
    # 5. COMPETING CANDIDATE ANALYSIS & 7. SCORE DECOMPOSITION
    # We will analyze candidates ranked ABOVE the lost GTs
    # However we didn't save `cand_obj` to save space. We can rebuild this using raw_dict
    
    # 9. COUNTERFACTUAL RANKING
    print("Running counterfactuals...")
    
    def rank_R0(c): # Production
        return (get_weight(c.channels, config), len(c.channels), c.matched_entity_id)
        
    def rank_R1(c): # sum(weights) only
        return (get_weight(c.channels, config), c.matched_entity_id)
        
    def rank_R2(c): # sum(weights) + count
        return (get_weight(c.channels, config) + len(c.channels), c.matched_entity_id)
        
    def rank_R3(c): # max weight
        max_w = max([config['channel_weights'].get(ch, 0.5) for ch in c.channels] + [0])
        return (max_w, c.matched_entity_id)
        
    def rank_R4(c): # lexicographic
        max_w = max([config['channel_weights'].get(ch, 0.5) for ch in c.channels] + [0])
        return (max_w, get_weight(c.channels, config), len(c.channels), c.matched_entity_id)
        
    def rank_R5(c): # strong count + weighted score
        s_count = sum(1 for ch in c.channels if is_strong(ch))
        return (s_count, get_weight(c.channels, config), c.matched_entity_id)
        
    rules = {'R0': rank_R0, 'R1': rank_R1, 'R2': rank_R2, 'R3': rank_R3, 'R4': rank_R4, 'R5': rank_R5}
    caps = [50, 100, 150, 200, 250, 500]
    
    results = []
    
    for r_name, r_func in rules.items():
        print(f"Evaluating {r_name}...")
        metrics = {'Rule': r_name}
        for cap in caps:
            retained = 0
            entities_lost_all = 0
            india_retained = 0
            us_retained = 0
            india_total = 0
            us_total = 0
            
            for s1, cands in raw_dict.items():
                gt = set(dev_gt.get(s1, []))
                ctry = country_map[s1]
                
                if ctry == 'India': india_total += len(gt)
                elif ctry == 'US': us_total += len(gt)
                
                # Sort
                sorted_cands = sorted(cands, key=r_func, reverse=True)
                capped = set(c.matched_entity_id for c in sorted_cands[:cap])
                
                found = len(gt & capped)
                retained += found
                
                if ctry == 'India': india_retained += found
                elif ctry == 'US': us_retained += found
                
                if len(gt & set(c.matched_entity_id for c in cands)) > 0 and found == 0:
                    entities_lost_all += 1
                    
            metrics[f'Cap_{cap}'] = retained / total_gt
            metrics[f'Cap_{cap}_LostAll'] = entities_lost_all
            metrics[f'Cap_{cap}_India'] = india_retained / max(1, india_total)
            metrics[f'Cap_{cap}_US'] = us_retained / max(1, us_total)
            
        results.append(metrics)
        
    pd.DataFrame(results).to_csv(ROOT / "output/candidate_ranking_counterfactuals.csv", index=False)
    print("Done analysis.")

if __name__ == "__main__":
    main()
